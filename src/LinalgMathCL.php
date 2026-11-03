<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\LinearBuffer;
use InvalidArgumentException;
use LogicException;

trait LinalgMathCL
{
    public function setSeed(int $seed) : void
    {
        $this->openclmath->pcg32Srand($this->rngState, $seed, self::DEFAULT_PCG32_SEQUENCE);
    }

    public function randInt(
        ?int $min=null,
        ?int $max=null,
        ?NDArray $out=null,
        ?object $events=null,
        ?object $waitEvents=null
    ) : NDArray
    {
        if($out===null) {
            $out = $this->alloc([],dtype:NDArray::int32);
        }
        $this->openclmath->pcg32RandInt32(
            $this->rngState,
            $min ?? -2147483648,
            $max ?? 2147483647,
            $out->buffer(),
            $out->offset(),
            $events,
            $waitEvents
        ); // range of int32
        return $out;
    }

    public function randInt64(
        ?int $min=null,
        ?int $max=null,
        ?NDArray $out=null,
        ?object $events=null,
        ?object $waitEvents=null
        ) : NDArray
    {
        if($out===null) {
            $out = $this->alloc([],dtype:NDArray::int64);
        }
        $this->openclmath->pcg32RandInt(
            $this->rngState,
            $min ?? -2147483648,
            $max ?? 2147483647,
            $out->buffer(),
            $out->offset(),
            $events,
            $waitEvents
        ); // range of int64
        return $out;
    }
    
    public function fill(
        mixed $value,
        array|NDArray $X,
        ?int $dtype=null,
        ?int $flags=null,
        ?object $events=null, ?object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("fill");
        }
        if(is_array($X)) {
            $X = $this->alloc($X,dtype:$dtype,flags:$flags);
        }
        if($this->isOpenCL110) {
            if(is_scalar($value)) {
                if(is_string($value)) {
                    $value = ord($value);
                }
            } elseif($value instanceof NDArray) {
                $pbuf = $value->buffer();
                if($value->size()!=1) {
                    throw new InvalidArgumentException('Value must be scalar');
                }
                $value = $pbuf[$value->offset()];
            } else {
                throw new InvalidArgumentException('Invalid value type');
            }
            $buffer = $X->buffer();
            $n = $X->size();
            $offsetX = $X->offset();
            $this->openclmath->fill(
                $n,$buffer,$offsetX,1,$value,
                $events,$waitEvents
            );
        } else {
            if(is_scalar($value)||$this->cisobject($value)) {
                if(is_string($value)) {
                    $value = ord($value);
                }
                $pattern = $this->allocHost([1],$X->dtype());
                $pattern[0] = $value;
            } elseif($value instanceof NDArray) {
                if($value->size()!=1) {
                    throw new InvalidArgumentException('Value must be scalar');
                }
                if(!($value->buffer() instanceof LinearBuffer)) {
                    throw new InvalidArgumentException('Value must have a host memory buffer. OpenCL Buffer is not allowed.');
                }
                $pattern = $value;
            } else {
                throw new InvalidArgumentException('Invalid data type');
            }
            $buffer = $X->buffer();
            $buffer->fill(
                $this->getQueue(),
                $pattern->buffer(),
                $X->size()*$buffer->valueSize(),
                $X->offset()*$X->valueSize(), // buffer offset
                $pattern->size(), // pattern size
                $pattern->offset(), // pattern offset
                $events,$waitEvents
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("fill");
        }
        return $X;
    }

    public function searchsorted(
        NDArray $A,
        NDArray $X,
        ?bool $right=null,
        ?int $dtype=null,
        ?NDArray $Y=null,
        ?object $events=null, ?object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("searchsorted");
        }
        if($A->ndim()==1) {
            $individual = false;
        } elseif($A->ndim()==2) {
            $individual = true;
        } else {
            throw new InvalidArgumentException('A must be 1D or 2D NDArray.');
        }
        if($right===null) {
            $right = false;
        }
        if($dtype===null) {
            $dtype = NDArray::int32;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),dtype:$dtype);
        }
        $dtype = $Y->dtype();
        if($dtype!=NDArray::uint32&&$dtype!=NDArray::int32&&
            $dtype!=NDArray::uint64&&$dtype!=NDArray::int64) {
            throw new InvalidArgumentException('dtype of Y must be int32 or int64');
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($individual) {
            [$m,$n] = $A->shape();
            if($m!=$X->size()) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension A,X: ".$shapeError);
            }
            $ldA = $n;
        } else {
            $m = $X->size();
            $n = $A->size();
            $ldA = 0;
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->openclmath->searchsorted(
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $right,
            $YY,$offY,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("searchsorted");
        }
        return $Y;
    }

    public function cumsum(
        NDArray $inputs,
        ?int $axis=null,
        ?bool $exclusive=null,
        ?bool $reverse=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("cumsum");
        }
        $ndim = $inputs->ndim();
        $origAxis = $axis;
        $axis ??= 0;
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $exclusive ??= false;
        $reverse ??= false;
        $postfixShape = $inputs->shape();
        $prefixShape = array_splice($postfixShape, 0, $axis);
        $m = array_product($prefixShape);
        $n = array_shift($postfixShape);
        $k = array_product($postfixShape);
        if($outputs===null) {
            $outputs = $this->alloc($inputs->shape(),dtype:$inputs->dtype());
        }
        if($inputs->shape()!=$outputs->shape()) {
            $shapeError = '('.implode(',',$inputs->shape()).'),('.implode(',',$outputs->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $AA = $inputs->buffer();
        $offA = $inputs->offset();
        $BB = $outputs->buffer();
        $offB = $outputs->offset();

        $this->openclmath->cumsumb(
            $m,
            $n,
            $k,
            $AA,$offA,
            $exclusive,
            $reverse,
            $BB,$offB,
            $events,$waitEvents,
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("cumsum");
        }
        return $outputs;
    }

    /**
     *     X := nan2num(X)
     */
    public function nan2num(
        NDArray $X,
        ?float $alpha=null,
        ?object $events=null,
        ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("nan2num");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 0.0;
        }

        $this->openclmath->nan2num(
            $n,
            $XX,$offX,1,
            $alpha,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("nan2num");
        }
        return $X;
    }

    /**
     *     X := isnan(X)
     */
    public function isnan(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("isnan");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->isnan(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("isnan");
        }
        return $X;
    }

    /**
     *     X := isfinite(X)
     */
    public function isfinite(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("isfinite");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->isfinite(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("isfinite");
        }
        return $X;
    }

    /**
     *     X := isinf(X)
     */
    public function isinf(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("isinf");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->isinf(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("isinf");
        }
        return $X;
    }

    //public function linspace(float $start, float $stop, int $num, ?int $dtype=null) : NDArray
    //{
    //    if($num<=0) {
    //        throw new InvalidArgumentException('num must be greater than or equal zero.');
    //    }
    //    $array = new NDArrayPhp(null,dtype:$dtype,shape:[$num],service:$this->service);
    //    $step = ($stop-$start)/($num-1);
    //    $value = $start;
    //    for($i=0;$i<$num;$i++) {
    //        $array[$i] = min($start+$step*$i,$stop);
    //    }
    //    $array = $this->array($array);
    //    return $array;
    //}

    public function linspace(
        float|array|NDArray $start,
        float|array|NDArray $stop,
        int $num,
        ?int $axis=null,
        ?int $dtype=null,
        ?NDArray $output=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("linspace");
        }
        if($num<=0) {
            throw new InvalidArgumentException('num must be greater than or equal zero.');
        }
        $axis ??= 0;
        if($dtype===null) {
            if($output!==null) {
                $dtype = $output->dtype();
            } elseif($start instanceof NDArray) {
                $dtype = $start->dtype();
            } else {
                $dtype ??= NDArray::float32;
            }
        }
        if(is_numeric($start)) {
            if(!is_numeric($stop)) {
                throw new InvalidArgumentException('if start is scalar, stop must be scalar.');
            }
            $start = $this->arrayHost($start,dtype:$dtype);
            $stop = $this->arrayHost($stop,dtype:$dtype);
        } else {
            if(is_numeric($stop)) {
                throw new InvalidArgumentException('if start is array, stop must be array.');
            }
            if(is_array($start)) {
                $start = $this->arrayHost($start,dtype:$dtype);
            }
            if(is_array($stop)) {
                $stop = $this->arrayHost($stop,dtype:$dtype);
            }
            if($start->shape() != $stop->shape()) {
                throw new InvalidArgumentException('start and stop must have same shape.');
            }
        }
        if($axis<0) {
            $axis = $start->ndim()+1+$axis;
        }
        $outershape = $start->shape();
        $innershape = array_splice($outershape,$axis);
        $shape = array_merge($outershape,[$num],$innershape);
        if($output!==null) {
            if($output->shape()!=$shape) {
                throw new InvalidArgumentException("output shape must be " . implode(',', $shape));
            }
        } else {
            $output = $this->alloc($shape,dtype:$dtype);
        }
        $outerStart = $start->reshape(
            [array_product($outershape),array_product($innershape)]);
        $outerStop = $stop->reshape(
            [array_product($outershape),array_product($innershape)]);
        $outerOutput = $output->reshape(
            [array_product($outershape),$num,array_product($innershape)]);
        $XX = $output->buffer();
        $incX = (int)array_product($innershape);
        foreach($outerStart as $outerIndex => $innerStart) {
            $innerStop = $outerStop[$outerIndex];
            $innerOutput = $outerOutput[$outerIndex];
            $offX = $innerOutput->offset();
            foreach($innerStart as $idx => $startValue) {
                $stopValue = $innerStop[$idx];
                if(!is_numeric($startValue)||!is_numeric($stopValue)) {
                    throw new InvalidArgumentException('start and stop must be numeric.');
                }
                $this->openclmath->linspace($num,$XX,$offX,$incX,$startValue,$stopValue,$events,$waitEvents);
                $offX += 1;
            }
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("linspace");
        }
        return $output;
    }

    public function range(
        int|float $limit,
        int|float|null $start=null,
        int|float|null $delta=null,
        ?int $dtype=null,
        ?NDArray $output=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("range");
        }
        $start ??= 0;
        $delta ??= ($limit>=$start)? 1 : -1;

        if($delta==0.0) {
            throw new InvalidArgumentException('infinite times');
        }
        $count = (int)ceil(($limit-$start)/$delta);

        if($output!=null) {
            if($output->shape()!=[$count]||$output->dtype()!=$dtype) {
                throw new InvalidArgumentException("output shape must be [" . $count . "]");
            }
        } else {
            $array = $this->alloc([$count],dtype:$dtype);
        }

        $XX = $array->buffer();
        $offsetX = $array->offset();
        $this->openclmath->range($count,$XX,$offsetX,1,$start,$limit,$delta,$events,$waitEvents);

        if($this->profiling) {
            $this->profilingEnd("range");
        }
        return $array;
    }

    public function astype(NDArray $X, int $dtype, ?NDArray $Y=null,
        ?object $events=null, ?object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("astype");
        }
        if($Y==null) {
            $Y = $this->alloc($X->shape(),dtype:$dtype);
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->openclmath->astype(
            $n,
            $dtype,
            $XX,$offX,1,
            $YY,$offY,1,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("astype");
        }
        return $Y;
    }

    /**
    *    ret := x_1 + ... + x_n
    */
    public function sum(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null, ?object $waitEvents=null,
        ) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("sum");
        }
        if($R==null) {
            $dtypeR = $X->dtype();
            if($dtypeR == NDArray::bool) {
                $dtypeR = NDArray::int32;
            }
            $R = $this->alloc([],dtype:$dtypeR,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $dtype = $X->dtype();
        //if($dtype==NDArray::float32 || $dtype==NDArray::float64) {
        //    $this->math->sum($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        //} else {
            $this->openclmath->sum($N,$RR,$offR,$XX,$offX,1,$events,$waitEvents);
        //}
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("sum");
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := arg max X(i)
    */
    public function imax(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null) : int|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("imax");
        }
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($this->isFloat($X)) {
            $this->math->imax($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        } else {
            $this->openclmath->reduceArgMax(
                1,          // $m
                $N,         // $n
                1,          // $k
                $XX, $offX,
                $RR, $offR,
                $events
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("imax");
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := arg min X(i)
    */
    public function imin(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null) : int|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("imin");
        }
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($this->isFloat($X)) {
            $this->math->imin($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        } else {
            $this->openclmath->imin($N,$RR,$offR,$XX,$offX,1,$events);
            //throw new InvalidArgumentException("Unsuppored data type. Only float types are supported.");
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        if($this->profiling) {
            $this->profilingEnd("imin");
        }
        return $R;
    }

    /**
    *    X := alpha * X + beta
    */
    public function increment(
        NDArray $X,
        ?float $beta=null,
        ?float $alpha=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("increment");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }

        $this->openclmath->increment(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("increment");
        }
        return $X;
    }

    /**
    *    X := 1 / (a*X + b)
    */
    public function reciprocal(
        NDArray $X,
        ?float $beta=null,
        ?float $alpha=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reciprocal");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }

        $this->openclmath->reciprocal(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reciprocal");
        }
        return $X;
    }

    /**
     * @return array<mixed>
     */
    protected function calcBroadcastFormat(NDArray $A,int|float|NDArray $X) : array
    {
        if(is_numeric($X)) {
            $X = new NDArrayPhp($X,dtype:$A->dtype(),service:$this->service);
            $X = $this->array($X);
        }
        if(!($X instanceof NDArray)) {
            throw new InvalidArgumentException('X must be NDArray or float');;
        }
        $ndimX = $X->ndim();
        $ndimA = $A->ndim();
        if($ndimX==0) {
            $X = $X->reshape([$X->size()]);
            $ndimX = 1;
            $size = $A->size();
            $A = $A->reshape([$size,1]);
            $ndimA = 2;
            $m = $size;
            $n = 1;
        } else {
            $shapeA = $A->shape();
            $shapeX = $X->shape();
            if($shapeA==$shapeX) {
                $m = 1;
                $n = $A->size();
            } else {
                $this->assertBroadcastShape($shapeA,$shapeX);
                $n = (int)array_product($shapeX);
                $m = (int)array_product(array_slice($shapeA, 0, -count($shapeX)));
            }
        }
        if($A->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException('A and X must be same data type');
        }
        $m = (int)$m;
        $n = (int)$n;
        return [$m,$n,$A,$X];
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] >  X[n])
     *     A[m,n] := X[n]   (A[m,n] <= X[n])
     */
    public function maximum(
        NDArray $A,
        int|float|NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("maximum");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->maximum(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("maximum");
        }
        return $A;
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] <  X[n])
     *     A[m,n] := X[n]   (A[m,n] >= X[n])
     */
    public function minimum(
        NDArray $A,
        int|float|NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("minimum");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->minimum(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("minimum");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] >  X[n])
     *     A[m,n] := 0 (A[m,n] <= X[n])
     */
    public function greater(
        NDArray $A,
        int|float|NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("greater");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->greater(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("greater");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] >= X[n])
     *     A[m,n] := 0 (A[m,n] <  X[n])
     */
    public function greaterEqual(
        NDArray $A,
        int|float|NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("greaterEqual");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->greaterEqual(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("greaterEqual");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] <  X[n])
     *     A[m,n] := 0 (A[m,n] >= X[n])
     */
    public function less(
        NDArray $A,
        int|float|NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("less");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->less(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("less");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] <= X[n])
     *     A[m,n] := 0 (A[m,n] >  X[n])
     */
    public function lessEqual(
        NDArray $A,
        int|float|NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("lessEqual");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->lessEqual(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("lessEqual");
        }
        return $A;
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
     public function multiply(
        NDArray $X,
        NDArray $A,
        ?bool $trans=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("multiply");
        }
        if($trans===null) {
            $trans = false;
        }
        $trans ??= false;
        $this->assertBroadcastShape($A->shape(),$X->shape(),trans:$trans);

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->openclmath->multiply(
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("multiply");
        }
        return $A;
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
    public function add(
        NDArray $X,
        NDArray $A,
        ?float $alpha=null,
        ?bool $trans=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("add");
        }
        $alpha ??= 1.0;
        $trans ??= false;
        $this->assertBroadcastShape($A->shape(),$X->shape(),trans:$trans);

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->openclmath->add(
            $trans,
            $m,
            $n,
            $alpha,
            $XX,$offX,1,
            $AA,$offA,$n,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("add");
        }
        return $A;
    }

    /**
     *     X := X ^ 2
     */
    public function square(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("square");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->square(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("square");
        }
        return $X;
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("sqrt");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->sqrt(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("sqrt");
        }
        return $X;
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        NDArray $X,
        ?float $beta=null,
        ?float $alpha=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rsqrt");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }

        $this->openclmath->rsqrt(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rsqrt");
        }
        return $X;
    }

    /**
     *     A(m,n) := A(m,n) ** alpha(n)
     */
    public function pow(
        NDArray $A,
        float|NDArray $alpha,
        ?bool $trans=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("pow");
        }
        $trans ??= false;
        $shapeA = $A->shape();
        if(is_numeric($alpha)) {
            $alpha = new NDArrayPhp($alpha,dtype:$A->dtype(),service:$this->service);
            $alpha = $this->array($alpha);
        }
        $shapeX = $alpha->shape();
        if(count($shapeX)==0) {
            $trans = false;
            $shapeA = [(int)array_product($shapeA),1];
            $shapeX = [1];
        }
        $this->assertBroadcastShape($shapeA,$shapeX,trans:$trans);

        $n = $alpha->size();
        $XX = $alpha->buffer();
        $offX = $alpha->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->openclmath->pow(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("pow");
        }
        return $A;
    }

    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("exp");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->exp(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("exp");
        }
        return $X;
    }

    /**
     *     X := log(X)
     */
    public function log(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("log");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->log(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("log");
        }
        return $X;
    }

    /**
     *     X := tanh(X)
     */
    public function tanh(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("tanh");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->tanh(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("tanh");
        }
        return $X;
    }

    /**
     *     X := sin(X)
     */
    public function sin(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("sin");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->sin(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("sin");
        }
        return $X;
    }

    /**
     *     X := cos(X)
     */
    public function cos(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("cos");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->cos(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("cos");
        }
        return $X;
    }

    /**
     *     X := tan(X)
     */
    public function tan(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("tan");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->tan(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("tan");
        }
        return $X;
    }

    /**
     *     Y(i) := 1 (X(i) = Y(i))
     *     Y(i) := 0 (X(i) != Y(i))
     */
    public function equal(
        NDArray $X,
        NDArray $Y,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("equal");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $incX = 1;
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $incY = 1;
        $this->openclmath->equal(
            $N,
            $XX,$offX,$incX,
            $YY,$offY,$incY,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("equal");
        }
        return $Y;
    }

    /**
     *     Y(i) := 1 (X(i) != Y(i))
     *     Y(i) := 0 (X(i) = Y(i))
     */
    public function notEqual(
        NDArray $X,
        NDArray $Y,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("notEqual");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $incX = 1;
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $incY = 1;
        $this->openclmath->notEqual(
            $N,
            $XX,$offX,$incX,
            $YY,$offY,$incY,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("notEqual");
        }
        return $Y;
    }

    /**
     *     X(i) := 1 (X(i)  = 0)
     *     X(i) := 0 (X(i) != 0)
     */
    public function not(
        NDArray $X,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("notEqual");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $incX = 1;
        $this->openclmath->not(
            $N,
            $XX,
            $offX,
            $incX,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("not");
        }
        return $X;
    }

    /**
     *      input:  X
     *      output: A
     *      A(m,n) := X(n)
     */
    public function duplicate(
        NDArray $input,
        ?int $repeats=null,
        ?bool $trans=null,
        ?NDArray $output=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("duplicate");
        }
        if($trans===null)
            $trans = false;
        if($output===null) {
            if($repeats===null)
                $repeats = 1;
            if(!$trans) {
                $output = $this->alloc(array_merge([$repeats],$input->shape()),dtype:$input->dtype());
            } else {
                $output = $this->alloc(array_merge($input->shape(),[$repeats]),dtype:$input->dtype());
            }
        } else {
            $this->assertBroadcastShape($output->shape(),$input->shape());
        }

        $n = $input->size();
        $XX = $input->buffer();
        $offX = $input->offset();
        $m = $output->size()/$n;
        $AA = $output->buffer();
        $offA = $output->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->openclmath->duplicate(
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("duplicate");
        }
        return $output;
    }

    /**
     * values, indices := topK(input,k,sorted=true)
     * 
     * @return array{NDArray,NDArray}
     */
    public function topK(
        NDArray $input,
        ?int $k=null,
        ?bool $sorted=null,
        ?NDArray $values=null,
        ?NDArray $indices=null,
        ?object $events=null, ?object $waitEvents=null
    ) : array
    {
        if($this->profiling) {
            $this->profilingStart("topK");
        }
        $k ??= 1;
        $sorted ??= true;

        $shapeInput = $input->shape();
        $shape = $shapeInput;
        $n = array_pop($shape);
        $m = (int)array_product($shape);
        array_push($shape,$k);
        if($k > $n) {
            throw new InvalidArgumentException('k must be less than or equal to the entity.');
        }

        if($values==null) {
            $values = $this->alloc($shape, dtype:$input->dtype());
        } else {
            if($values->shape()!=$shape) {
                $error = '['.implode(',',$values->shape()).']';
                throw new InvalidArgumentException('Unmatch shape of "values" with "input" and "k": '.$error);
            }
            if($values->dtype()!=$input->dtype()) {
                throw new InvalidArgumentException('Unmatch dtype of "values" with input');
            }
        }

        if($indices==null) {
            $indices = $this->alloc($shape, dtype:NDArray::int32);
        } else {
            if($indices->ndim()!=2) {
                throw new InvalidArgumentException('"indices" must be 2-D dimension');
            }
            if($indices->shape()!=$shape) {
                $error = '['.implode(',',$indices->shape()).']';
                throw new InvalidArgumentException('Unmatch shape of "indices" with "input" and "k": '.$error);
            }
            if($indices->dtype()!=NDArray::int32) {
                throw new InvalidArgumentException('Unmatch dtype of "indices" with input');
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $values->buffer();
        $offB = $values->offset();
        $XX = $indices->buffer();
        $offX = $indices->offset();

        // Call the underlying top_k function
        $this->openclmath->topK(
            $m,
            $n,
            $AA, $offA,
            $k,
            $sorted,
            $BB, $offB,
            $XX, $offX,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("topK");
        }
        return [$values, $indices];
    }

    /**
     *      input:  A,X
     *      output: B
     *      B(m,n,k) := A(m,X(m,n),k)
     */
    public function doGather(
        bool $scatterAdd,
        NDArray $A,
        NDArray $X,
        ?int $axis=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            $numClass = array_shift($postfixShape);
            $m = 1;
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $reductionDims = false;
            $outputShape = array_merge($prefixShape,$postfixShape);
        } else {
            $ndim = $A->ndim();
            $orgAxis = $axis;
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            $postfixShape = $A->shape();
            //$prefixShape = [];
            //for($i=0;$i<$axis;$i++) {
            //    $prefixShape[] = array_shift($postfixShape);
            //}
            $prefixShape = array_splice($postfixShape,0,$axis);
            $m = array_product($prefixShape);
            $numClass = array_shift($postfixShape);
            $n = array_product($postfixShape);
            $k = 1;
            $reductionDims = true;
            $outputShape = array_merge($prefixShape,$postfixShape);
            if($X->shape()!=$outputShape) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$A,$X]));
            }
        }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($output, events:$waitEvents, waitEvents:$waitPrev);
        } else {
            if($output->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape of dimension: ".
                                            $this->printableShapes([$outputShape,$output]));
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        if($scatterAdd) {
            $reverse=true;
            $addMode=true;
        } else {
            $reverse=false;
            $addMode=false;
        }
        if($reductionDims) {
            $this->openclmath->reduceGather(
                $reverse,
                $addMode,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $AA,$offA,
                $BB,$offB,
                $events, $waitEvents
            );
        } else {
            if($scatterAdd) {
                $this->openclmath->scatterAdd(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
            } else {
                $this->openclmath->gather(
                    $reverse,
                    $addMode,
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
            }
        }
        if($this->blocking) {
            $this->finish();
        }
        return $output;
    }

    /**
    *      B(m,n,k) := A(m,X(m,n),k)
    */
    public function gather(
        NDArray $A,
        NDArray $X,
        ?int $axis=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gather");
        }
        $output = $this->doGather(
            $scatterAdd=false,
            $A,
            $X,
            $axis,
            $output,
            $dtype,
            $events, $waitEvents
        );
        if($this->profiling) {
            $this->profilingEnd("gather");
        }
        return $output;
    }

    /**
     *      input:  A,X
     *      output: B
     *      B(m,X(m,n),k) += A(m,n,k)
     */
    public function scatterAdd(
        NDArray $X,
        NDArray $A,
        NDArray $output,
        ?int $axis=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatterAdd");
        }
        $this->doGather(
            $scatterAdd=true,
            $output,
            $X,
            $axis,
            $A,
            $dtype,
            $events, $waitEvents
        );
        if($this->profiling) {
            $this->profilingEnd("scatterAdd");
        }
        return $output;
    }

    /**
     *      input:  A,X
     *      output: B
     *      B(m,X(m,n),k) := A(m,n,k)
     */
    public function scatter(
        NDArray $X,
        NDArray $A,
        int $numClass,
        ?int $axis=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatter");
        }
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
//echo "axis=$axis,numClass=$numClass\n";
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            //$numClass
            //$ndimX = $X->ndim();
            //$tmpShape = [];
            //for($i=0;$i<$ndimX;$i++) {
            //    $tmpShape[] = array_shift($postfixShape);
            //}
            $tmpShape = array_splice($postfixShape,0,$X->ndim());
            if($tmpShape!=$prefixShape) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$X,$A]));
            }
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $m = 1;
            $expandDims = false;
            $outputShape = array_merge([$numClass],$postfixShape);
        } else {
            $ndim = $A->ndim();
            $orgAxis = $axis;
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            //if($axis<0 || $axis>$ndim-1) {
            //    throw new InvalidArgumentException("Invalid axis: ".$orgAxis);
            //}
            $postfixShape = $A->shape();
            $postfixX = $X->shape();
            if($postfixShape!=$postfixX) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$X,$A]));
            }
            //$prefixShape = [];
            //for($i=0;$i<$axis;$i++) {
            //    $prefixShape[] = array_shift($postfixShape);
            //    array_shift($postfixX);
            //}
            $prefixShape = array_splice($postfixShape,0,$axis);
            $m = array_product($prefixShape);
            $n = array_product($postfixShape);
            $k = 1;
            $expandDims = true;
            $outputShape = array_merge($prefixShape,[$numClass],$postfixShape);
        }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($output, events:$waitEvents, waitEvents:$waitPrev);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        if($expandDims) {
            $this->openclmath->reduceGather(
                $reverse=true,
                $addMode=false,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA,
                $events, $waitEvents
            );
        } else {
            $this->openclmath->gather(
                $reverse=true,
                $addMode=false,
                $n,
                $k,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA,
                $events, $waitEvents
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scatter");
        }
        return $output;
    }

    /**
     * params:  (batches, m, numClass, k, len)
     * indices: (batches, n, k)
     * outputs: (batches, m, n, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
     */
    public function doGatherb(
        bool $reverse,
        bool $addMode,
        NDArray $A,
        NDArray $X,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $B=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        $batchDims ??= 0;
        if($batchDims<0) {
            $batchDims += $A->ndim();
        }
        if($batchDims>=$A->ndim()) {
            throw new InvalidArgumentException(
                "batchDims ($batchDims) must be less than to ndims of A (".$A->ndim().")"
            );
        }
        $axis ??= $batchDims;
        if($axis<0) {
            $axis += $A->ndim();
        }
        if($axis>=$A->ndim()) {
            throw new InvalidArgumentException(
                "axis ($axis) must be less than to ndims of A (".$A->ndim().")"
            );
        }
        $detailDepth ??= $axis+1;
        $indexDepth ??= $X->ndim();
        if($batchDims>$axis) {
            if($axis==0) {
                $batchDims = 0;
            } else {
                throw new InvalidArgumentException("batchDims ($batchDims) must be less than or equal to axis ($axis)");
            }
        }

        //echo "params  sh: ".$this->printableShapes($A->shape())."\n";
        //echo "indices sh: ".$this->printableShapes($X->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        // A
        $batchShape = $A->shape();
        $outerShape = array_splice($batchShape, $batchDims);
        $innerShape = array_splice($outerShape, $axis-$batchDims);
        $numClass = array_shift($innerShape);
        $detailShape = array_splice($innerShape, $detailDepth-1-$axis);
        // X
        $batchShapeX = $X->shape();
        $indexShape = array_splice($batchShapeX, $batchDims);
        $innerShapeX = array_splice($indexShape, $indexDepth-$batchDims);

        //echo "==========\n";
        //echo "params:     ".$this->printableShapes($A->shape())."\n";
        //echo "indices:    ".$this->printableShapes($X->shape())."\n";
        //echo "batchDims:  $batchDims\n";
        //echo "axis:       $axis\n";
        //echo "detailDepth:$detailDepth\n";
        //echo "indexDepth: $indexDepth\n";
        //echo "==params==\n";
        //echo "batchShape: ".$this->printableShapes($batchShape)."\n";
        //echo "outerShape: ".$this->printableShapes($outerShape)."\n";
        //echo "numClass:   ".$numClass."\n";
        //echo "innerShape: ".$this->printableShapes($innerShape)."\n";
        //echo "detailShape: ".$this->printableShapes($detailShape)."\n";
        //echo "==indices==\n";
        //echo "batchShape: ".$this->printableShapes($batchShapeX)."\n";
        //echo "indexShape: ".$this->printableShapes($indexShape)."\n";
        //echo "innerShape: ".$this->printableShapes($innerShapeX)."\n";
        if($batchShape!=$batchShapeX) {
            throw new InvalidArgumentException(
                "Unmatch batch shape of params and indices: ".
                $this->printableShapes($batchShape).",".
                $this->printableShapes($batchShapeX).","
            );
        }
        if($innerShape!=$innerShapeX) {
            throw new InvalidArgumentException(
                "Unmatch inner shape of params and indices: ".
                "param's inner shape is ".$this->printableShapes($innerShape).",".
                "index's inner shape is ".$this->printableShapes($innerShapeX).","
            );
        }

        $batches = array_product($batchShape);
        $m = array_product($outerShape);
        $n = array_product($indexShape);
        $k = array_product($innerShape);
        $len = array_product($detailShape);
        $outputShape = array_merge($batchShape, $outerShape, $indexShape, $innerShape, $detailShape);
        //echo "==outputs==\n";
        //echo "outputShape: ".$this->printableShapes($outputShape)."\n";
        //echo "batches=$batches,m=$m,n=$n,k=$k,len=$len,numClass=$numClass\n";
        if($B==null) {
            $B = $this->alloc($outputShape, dtype:$A->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($B, events:$waitEvents, waitEvents:$waitPrev);
        } else {
            if($B->shape()!=$outputShape) {
                $expects = $this->printableShapes($outputShape);
                $gives = $this->printableShapes($B->shape());
                if($reverse) {
                    $msg = "Unmatch updates shape: expects {$expects}, but gives {$gives}";
                } else {
                    $msg = "Unmatch output shape: expects {$expects}, but gives {$gives}";
                }
                throw new InvalidArgumentException($msg);
            }
        }
        $AA = $A->buffer();
        $offsetA = $A->offset();
        $XX = $X->buffer();
        $offsetX = $X->offset();
        $BB = $B->buffer();
        $offsetB = $B->offset();
        $this->openclmath->gatherb(
            $reverse,
            $addMode,
            $batches,
            $m,
            $n,
            $k,
            $len,
            $numClass,
            $AA,
            $offsetA,
            $XX,
            $offsetX,
            $BB,
            $offsetB,
        );
        return $B;
    }

    /**
     * params: (batches, m, numClass, k, len)
     * indices: (batches, n, k)
     * outputs: (batches, m, n, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
     */
    public function gatherb(
        NDArray $params,
        NDArray $indices,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gatherb");
        }
        $results = $this->doGatherb(
            $reverse=false,
            $addMode=false,
            $params,
            $indices,
            $axis,
            $batchDims,
            $detailDepth,
            $indexDepth,
            $outputs,
            $events, $waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("gatherb");
        }
        return $results;
    }

    /**
     * indices: (batches, n, k)
     * updates: (batches, m, n, k, len)
     * outputs: (batches, m, numClass, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
     * @param array<int> $shape
    */
    public function scatterb(
        NDArray $indices,
        NDArray $updates,
        array $shape,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
    ): NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatterb");
        }
        if($outputs==null) {
            $outputs = $this->alloc($shape,$updates->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($outputs, events:$waitEvents, waitEvents:$waitPrev);
        }
        $this->doGatherb(
            $reverse=true,
            $addMode=false,
            $outputs,
            $indices,
            $axis,
            $batchDims,
            $detailDepth,
            $indexDepth,
            $updates,
            $events, $waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scatterb");
        }
        return $outputs;
    }

    /**
     * indices: (batches, n, k)
     * updates: (batches, m, n, k, len)
     * outputs: (batches, m, numClass, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
     * @param array<int> $shape
    */
    public function scatterbAdd(
        NDArray $indices,
        NDArray $updates,
        array $shape,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
    ): NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatterbAdd");
        }
        if($outputs==null) {
            $outputs = $this->alloc($shape,$updates->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($outputs, events:$waitEvents, waitEvents:$waitPrev);
        }
        $this->doGatherb(
            $reverse=true,
            $addMode=true,
            $outputs,
            $indices,
            $axis,
            $batchDims,
            $detailDepth,
            $indexDepth,
            $updates,
            $events, $waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scatterbAdd");
        }
        return $outputs;
    }

    /**
     * params:  (m, (indices(m,n)), k)
     * indices: (m, n, index_depth)
     * outputs: (m, n, k)
     */
    public function doGatherND(
        bool $reverse,
        bool $addMode,
        NDArray $A, 
        NDArray $X,
        ?int $batchDims=null,
        ?NDArray $B=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        $batchDims ??= 0;
        //echo "params  sh: ".$this->printableShapes($A->shape())."\n";
        //echo "indices sh: ".$this->printableShapes($X->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        $batchShape = $X->shape();
        $indexDepth = array_pop($batchShape);
        $outerShape = array_splice($batchShape,$batchDims);

        $innerShape = $A->shape();
        $paramsBatchShape = array_splice($innerShape,0,$batchDims);
        if($paramsBatchShape!=$batchShape) {
            throw new InvalidArgumentException("Unmatch batch shape of params and indices: ".
                $this->printableShapes([$paramsBatchShape,$batchShape]));
        }
        if($indexDepth > $A->ndim()-$batchDims) {
            throw new InvalidArgumentException(
                "ndim of params must greater than index depth or equal: ".
                "ndim of params gives ".$A->ndim()." , index depth gives ".$indexDepth);
        }
        $paramShape = array_splice($innerShape,0,$indexDepth);
        $outputShape = array_merge($batchShape,$outerShape,$innerShape);
        $m = array_product($batchShape);
        $n = array_product($outerShape);
        $k = array_product($innerShape);
        //echo "batchShape: ".$this->printableShapes($batchShape)."\n";
        //echo "outerShape: ".$this->printableShapes($outerShape)."\n";
        //echo "innerShape: ".$this->printableShapes($innerShape)."\n";
        //echo "indexDepth: ".$indexDepth."\n";
        //echo "outputShape: ".$this->printableShapes($outputShape)."\n";
        if($B==null) {
            $B = $this->alloc($outputShape,dtype:$A->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($B, events:$waitEvents, waitEvents:$waitPrev);
        } else {
            if($B->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape: ".
                    $this->printableShapes([$outputShape]));
            }
        }
        $paramSize = (int)array_product($paramShape);
        $paramShape = new NDArrayPhp($paramShape,dtype:NDArray::int32,service:$this->service);
        $paramShape = $paramShape->buffer();
        $AA = $A->buffer();
        $offsetA = $A->offset();
        $XX = $X->buffer();
        $offsetX = $X->offset();
        $BB = $B->buffer();
        $offsetB = $B->offset();
        $this->openclmath->gathernd(
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $indexDepth,
            $paramShape, // paramShape[indexDepth]
            $AA, $offsetA,
            $XX, $offsetX,
            $BB, $offsetB,
            $events, $waitEvents
        );
        return $B;
    }

    /**
     * This function is unofficial.
     * It may be removed or changed without notice.
     * 
     * params:  (m, (indices(m,n)), k)
     * indices: (m, n, index_depth)
     * outputs: (m, n, k)
     */
    public function gatherND(
        NDArray $params, 
        NDArray $indices,
        ?int $batchDims=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gatherND");
        }
        $results = $this->doGatherND(
            $reverse=false,
            $addMode=false,
            $params,
            $indices,
            $batchDims,
            $outputs,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("gatherND");
        }
        return $results;
    }

    /**
     * This function is unofficial.
     * It may be removed or changed without notice.
     * 
     * params:  (m, n, k)
     * indices: (m, n, index_depth)
     * outputs: (m, (indices(m,n)), k)
     * @param array<int> $shape
     */
    public function scatterND(
        NDArray $indices,
        NDArray $updates,
        array $shape,
        ?int $batchDims=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatterND");
        }
        if($outputs==null) {
            $outputs = $this->alloc($shape,$updates->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($outputs, events:$waitEvents, waitEvents:$waitPrev);
        }
        
        $this->doGatherND(
            $reverse=true,
            $addMode=false,
            $outputs,
            $indices,
            $batchDims,
            $updates,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scatterND");
        }
        return $outputs;
    }

    /**
     * This function is unofficial.
     * It may be removed or changed without notice.
     * 
     * params:  (m, n, k)
     * indices: (m, n, index_depth)
     * outputs: (m, (indices(m,n)), k)
     * @param array<int> $shape
     */
    public function scatterNDAdd(
        NDArray $indices,
        NDArray $updates,
        array $shape,
        ?int $batchDims=null,
        ?NDArray $outputs=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatterNDAdd");
        }
        if($outputs==null) {
            $outputs = $this->alloc($shape,$updates->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($outputs, events:$waitEvents, waitEvents:$waitPrev);
        }
        
        $this->doGatherND(
            $reverse=true,
            $addMode=true,
            $outputs,
            $indices,
            $batchDims,
            $updates,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scatterNDAdd");
        }
        return $outputs;
    }

    public function onehot(
        NDArray $X,
        int $numClass,
        ?float $alpha=null,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("onehot");
        }
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        $sizeX = $X->size();
        $addMode = true;
        if($output===null) {
            $addMode = false;
            $output = $this->alloc([$sizeX,$numClass],dtype:$this->defaultFloatType);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($output, events:$waitEvents, waitEvents:$waitPrev);
        }
        if($output->ndim()!=2) {
            throw new InvalidArgumentException('"Y" must be 2D-NDArray.');
        }
        [$m,$n] = $output->shape();
        if($m!=$sizeX || $n!=$numClass) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$output->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        if($alpha===null) {
            $alpha = 1.0;
        }
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $output->buffer();
        $offY = $output->offset();
        $ldY = $n;

        $this->openclmath->onehot(
            $m,
            $n,
            $alpha,
            $XX,$offX,1,
            $YY,$offY,$ldY,
            $addMode,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("onehot");
        }
        return $output;
    }

    /**
     *     X := softmax(X)
     */
    public function softmax(
        NDArray $X,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("softmax");
        }
        if($X->ndim()!=2) {
            throw new InvalidArgumentException('"X" must be 2-D dimension');
        }

        [$m,$n] = $X->shape();
        $XX = $X->buffer();
        $offX = $X->offset();
        $ldA = $n;
        $this->openclmath->softmax(
            $m,
            $n,
            $XX,$offX,$ldA,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("softmax");
        }
        return $X;
    }

    /**
     *    X(m) := sum( A(m,n) )
     */
    public function reduceSum(//reduceSumEx
        NDArray $input,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reduceSum");
        }
        $ndim = $input->ndim();
        $origAxis = $axis;
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $postfixShape = $input->shape();
        //$prefixShape = [];
        //for($i=0;$i<$axis;$i++) {
        //    $prefixShape[] = array_shift($postfixShape);
        //}
        $prefixShape = array_splice($postfixShape,0,$axis);
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        if($keepdims) {
            $outputShape = array_merge($prefixShape,[1],$postfixShape);
        } else {
            $outputShape = array_merge($prefixShape,$postfixShape);
        }
        if($dtype===null) {
            $dtype = $input->dtype();
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$input->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        $this->openclmath->reduceSum(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reduceSum");
        }
        return $output;
    }

    /**
     *    X(m) := max( A(m,n) )
     */
    public function reduceMax(//reduceMaxEx
        NDArray $input,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reduceMax");
        }
        $ndim = $input->ndim();
        $origAxis = $axis;
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $postfixShape = $input->shape();
        //$prefixShape = [];
        //for($i=0;$i<$axis;$i++) {
        //    $prefixShape[] = array_shift($postfixShape);
        //}
        $prefixShape = array_splice($postfixShape,0,$axis);
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        if($keepdims) {
            $outputShape = array_merge($prefixShape,[1],$postfixShape);
        } else {
            $outputShape = array_merge($prefixShape,$postfixShape);
        }
        if($dtype===null) {
            $dtype = $input->dtype();
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$input->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        $this->openclmath->reduceMax(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reduceMax");
        }
        return $output;
    }

    /**
     *    X(m) := imax( A(m,n) )
     */
    public function reduceArgMax(//reduceArgMaxEx
        NDArray $input,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reduceArgMax");
        }
        $ndim = $input->ndim();
        $origAxis = $axis;
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $postfixShape = $input->shape();
        //$prefixShape = [];
        //for($i=0;$i<$axis;$i++) {
        //    $prefixShape[] = array_shift($postfixShape);
        //}
        $prefixShape = array_splice($postfixShape,0,$axis);
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        if($keepdims) {
            $outputShape = array_merge($prefixShape,[1],$postfixShape);
        } else {
            $outputShape = array_merge($prefixShape,$postfixShape);
        }
        if($dtype===null) {
            $dtype = NDArray::int32;
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$input->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        $this->openclmath->reduceArgMax(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reduceArgMax");
        }
        return $output;
    }

    /**
     *    X(m) := sum( A(m,n)/n )
     */
    public function reduceMean(
        NDArray $input,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?NDArray $output=null,
        ?int $dtype=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        //$waitPrev = $waitEvents;
        //$waitEvents = $this->newEventList();
        if($axis===null) {
            $axis = 0;
        }
        $output = $this->reduceSum(
            $input,axis:$axis,keepdims:$keepdims,output:$output,dtype:$dtype,
            /*events:$waitEvents,waitEvents:$waitPrev*/
            waitEvents:$waitEvents
        );
        $ndim = $input->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($ndim<=$axis) {
            throw new InvalidArgumentException('axis must be less then num of dimension');
        }
        $shapeA = $input->shape();
        $rows = $shapeA[$axis];
        //$waitEvents->wait();
        $this->scal(
            1/$rows,$output,
            $events
        );
        if($this->blocking) {
            $this->finish();
        }
        return $output;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     * @param bool|array<int> $padding
     */
    public function im2col2dclblast(
        bool $reverse,
        int $kernel_mode,
        NDArray $images,
        ?array $filterSize=null,
        ?array $strides=null,
        bool|array|null $padding=null,
        ?array $dilation_rate=null,
        //?bool $channels_first=null,
        //?bool $cols_channels_first=null,
        ?NDArray $cols=null,
        ?object $events=null
        ) : NDArray
    {
        if($this->openclmath->hasDiv5Bug()) {
            throw new LogicException('Not support function on this device.');
        }

        if($this->profiling) {
            $this->profilingStart("im2col2dclblast");
        }
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=4) {
            throw new InvalidArgumentException('images must be 4D dimension');
        }
        //if($channels_first) {
            [$batches,
             $channels,
             $im_h,$im_w] =
                $images->shape();
        //} else {
        //    [$batches,
        //     $im_h,$im_w,
        //     $channels] =
        //        $images->shape();
        //}
        if($filterSize==null) {
            $filterSize = [3,3];
        }
        [$filter_h,$filter_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1,1];
        }
        [$stride_h,$stride_w] =
            $strides;
        if($padding===null) {
            $padding = false;
        }
        if($dilation_rate==null) {
            $dilation_rate = [1,1];
        }
        [$dilation_h,$dilation_w] =
            $dilation_rate;
        //$channels_first = ($channels_first) ? true:false;
        //$cols_channels_first = ($cols_channels_first) ? true:false;
        $out_h = intdiv(($im_h-($filter_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($filter_w-1)*$dilation_w-1),$stride_w)+1;
        if($out_h<=0 && $out_w<=0) {
            throw new InvalidArgumentException('Invalid shape or parameters.');
        }
        if(is_bool($padding)) {
            if($padding) {
                $out_h = $im_h;
                $out_w = $im_w;
                $padding_h = (int)(($im_h-1)*$stride_h-$im_h+($filter_h-1)*$dilation_h+1);
                if($padding_h%2) {
                    $out_h++;
                }
                $padding_h = $padding_h ? intdiv($padding_h,2) : 0;
                $padding_w = (int)(($im_w-1)*$stride_w-$im_w+($filter_w-1)*$dilation_w+1);
                if($padding_w%2) {
                    $out_w++;
                }
                $padding_w = $padding_w ? intdiv($padding_w,2) : 0;
            } else {
                $padding_h = 0;
                $padding_w = 0;
            }
        } elseif(is_array($padding)) {
            $padding_h = $padding[0];
            $padding_w = $padding[1];
            $out_h = $im_h + $padding_h;
            $out_w = $im_w + $padding_w;
        } else {
            throw new InvalidArgumentException('padding must be bool or array');
        }
        if($cols==null) {
            //if($cols_channels_first) {
                //echo "($batches,$channels,$filter_h,$filter_w,$out_h,$out_w)\n";
                $cols = $this->alloc([
                    $batches,$channels,$filter_h,$filter_w, // channels_first
                    $out_h,$out_w,                          // filters first
                ],dtype:$images->dtype());
                //$zerosEvents = $this->newEventList();
                $this->zeros($cols,/*events:$zerosEvents*/);
                //$zerosEvents->wait();
            //} else {
            //    $cols = $this->alloc([
            //        $batches,$out_h,$out_w,
            //        $filter_h,$filter_w,$channels
            //    ]);
            //    $this->zeros($cols);
            //}
        }

        $col_buffer = $cols->buffer();
        $col_offset = $cols->offset();
        //$out_size = $cols->size();
        $image_step = $channels*$im_h*$im_w;
        $col_step   = $channels*$filter_h*$filter_w*$out_h*$out_w;
        if(!$reverse) {
            for($i=0;$i<$batches;$i++,$images_offset+=$image_step,$col_offset+=$col_step) {
                $this->math->im2col(
                    $kernel_mode,
                    $channels,
                    $im_h,
                    $im_w,
                    $filter_h,
                    $filter_w,
                    $padding_h,
                    $padding_w,
                    $stride_h,
                    $stride_w,
                    $dilation_h,
                    $dilation_w,
                    $images_buff,
                    $images_offset,
                    $col_buffer,
                    $col_offset,
                    $this->queue,
                    //$batches,
                    //$channels_first,
                    //$cols_channels_first,
                    $events
                );
            }
            $result = $cols;
        }  else {
            for($i=0;$i<$batches;$i++,$images_offset+=$image_step,$col_offset+=$col_step) {
                $this->math->col2im(
                    $kernel_mode,
                    $channels,
                    $im_h,
                    $im_w,
                    $filter_h,
                    $filter_w,
                    $padding_h,
                    $padding_w,
                    $stride_h,
                    $stride_w,
                    $dilation_h,
                    $dilation_w,
                    $col_buffer,
                    $col_offset,
                    $images_buff,
                    $images_offset,
                    $this->queue,
                    //$batches,
                    //$channels_first,
                    //$cols_channels_first,
                    $events
                );
            }
            $result = $images;
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("im2col2dclblast");
        }
        return $result;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function im2col(
        NDArray $images,
        ?array $filterSize=null,
        ?array $strides=null,
        ?bool $padding=null,
        ?bool $channels_first=null,
        ?array $dilation_rate=null,
        ?bool $cols_channels_first=null,
        ?NDArray $cols=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("im2col");
        }
        $ndim = $images->ndim();
        if($ndim==3) {
            $_cols = $this->im2col1d(
                false,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $dilation_rate,
                $cols_channels_first,
                $cols,
                $events,$waitEvents
            );
            if($cols==null) {
                $cols = $_cols;
            }
        } elseif($ndim==4) {
            $_cols = $this->im2col2d(
                false,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $dilation_rate,
                $cols_channels_first,
                $cols,
                $events,$waitEvents
            );
            if($cols==null) {
                $cols = $_cols;
            }
        } elseif($ndim==5) {
            $_cols = $this->im2col3d(
                false,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $dilation_rate,
                $cols_channels_first,
                $cols,
                $events,$waitEvents
            );
            if($cols==null) {
                $cols = $_cols;
            }
        } else {
            throw new InvalidArgumentException('unsuppoted images shape');
        }
        if($this->profiling) {
            $this->profilingEnd("im2col");
        }
        return $cols;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function col2im(
        NDArray $cols,
        NDArray $images,
        ?array $filterSize=null,
        ?array $strides=null,
        ?bool $padding=null,
        ?bool $channels_first=null,
        ?array $dilation_rate=null,
        ?bool $cols_channels_first=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("col2im");
        }
        $ndim = $images->ndim();
        if($ndim==3) {
            $this->im2col1d(
                true,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $dilation_rate,
                $cols_channels_first,
                $cols,
                $events,$waitEvents
            );
        } elseif($ndim==4) {
            $this->im2col2d(
                true,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $dilation_rate,
                $cols_channels_first,
                $cols,
                $events,$waitEvents
            );
        } elseif($ndim==5) {
            $this->im2col3d(
                true,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $dilation_rate,
                $cols_channels_first,
                $cols,
                $events,$waitEvents
            );
        } else {
            throw new InvalidArgumentException('unsuppoted images shape');
        }
        if($this->profiling) {
            $this->profilingEnd("col2im");
        }
        return $images;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function im2col1d(
        bool $reverse,
        NDArray $images,
        ?array $filterSize=null,
        ?array $strides=null,
        ?bool $padding=null,
        ?bool $channels_first=null,
        ?array $dilation_rate=null,
        ?bool $cols_channels_first=null,
        ?NDArray $cols=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=3) {
            throw new InvalidArgumentException('images must be 3D dimension');
        }
        if($channels_first) {
            [$batches,
             $channels,
             $im_w] =
                $images->shape();
        } else {
            [$batches,
             $im_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3];
        }
        [$kernel_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1];
        }
        [$stride_w] =
            $strides;
        $padding = ($padding) ? true:false;
        $channels_first = ($channels_first) ? true:false;
        if($dilation_rate==null) {
            $dilation_rate = [1];
        }
        [$dilation_w] =
            $dilation_rate;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_w = $im_w;
            } else {
                $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
            }
            if($out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $channels,$kernel_w
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols, events:$waitEvents, waitEvents:$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $kernel_w,$channels
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols, events:$waitEvents, waitEvents:$waitPrev);
            }
        }
        $cols_buffer = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $this->openclmath->im2col1d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_w,
            $channels,
            $kernel_w,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_w,
            $cols_channels_first,
            $cols_buffer,
            $cols_offset,
            $cols_size,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $cols;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function im2col2d(
        bool $reverse,
        NDArray $images,
        ?array $filterSize=null,
        ?array $strides=null,
        ?bool $padding=null,
        ?bool $channels_first=null,
        ?array $dilation_rate=null,
        ?bool $cols_channels_first=null,
        ?NDArray $cols=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=4) {
            throw new InvalidArgumentException('images must be 4D dimension');
        }
        if($channels_first) {
            [$batches,
             $channels,
             $im_h,$im_w] =
                $images->shape();
        } else {
            [$batches,
             $im_h,$im_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3,3];
        }
        [$kernel_h,$kernel_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1,1];
        }
        [$stride_h,$stride_w] =
            $strides;
        $padding = ($padding) ? true:false;
        $channels_first = ($channels_first) ? true:false;
        if($dilation_rate==null) {
            $dilation_rate = [1,1];
        }
        [$dilation_h,$dilation_w] =
            $dilation_rate;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_h = $im_h;
                $out_w = $im_w;
            } else {
                $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
                $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
            }
            if($out_h<=0 && $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or parameters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $channels,$kernel_h,$kernel_w
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols, events:$waitEvents, waitEvents:$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $kernel_h,$kernel_w,$channels
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols, events:$waitEvents, waitEvents:$waitPrev);
            }
        }
        $cols_buffer = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $this->openclmath->im2col2d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_h,
            $im_w,
            $channels,
            $kernel_h,
            $kernel_w,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_h,
            $dilation_w,
            $cols_channels_first,
            $cols_buffer,
            $cols_offset,
            $cols_size,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $cols;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function im2col3d(
        bool $reverse,
        NDArray $images,
        ?array $filterSize=null,
        ?array $strides=null,
        ?bool $padding=null,
        ?bool $channels_first=null,
        ?array $dilation_rate=null,
        ?bool $cols_channels_first=null,
        ?NDArray $cols=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=5) {
            throw new InvalidArgumentException('images must be 5D dimension');
        }
        if($channels_first) {
            [$batches,
             $channels,
             $im_d,$im_h,$im_w] =
                $images->shape();
        } else {
            [$batches,
             $im_d,$im_h,$im_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3,3,3];
        }
        [$kernel_d,$kernel_h,$kernel_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1,1,1];
        }
        [$stride_d,$stride_h,$stride_w] =
            $strides;
        $padding = ($padding) ? true:false;
        $channels_first = ($channels_first) ? true:false;
        if($dilation_rate==null) {
            $dilation_rate = [1,1,1];
        }
        [$dilation_d,$dilation_h,$dilation_w] =
            $dilation_rate;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_d = $im_d;
                $out_h = $im_h;
                $out_w = $im_w;
            } else {
                $out_d = intdiv(($im_d-($kernel_d-1)*$dilation_d-1),$stride_d)+1;
                $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
                $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
            }
            if($out_d<=0 || $out_h<=0 || $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $channels,$kernel_d,$kernel_h,$kernel_w
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols, events:$waitEvents, waitEvents:$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $kernel_d,$kernel_h,$kernel_w,$channels
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols, events:$waitEvents, waitEvents:$waitPrev);
            }
        }
        $cols_buffer = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $this->openclmath->im2col3d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels,
            $kernel_d,
            $kernel_h,
            $kernel_w,
            $stride_d,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_d,
            $dilation_h,
            $dilation_w,
            $cols_channels_first,
            $cols_buffer,
            $cols_offset,
            $cols_size,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $cols;
    }

    /**
    *  random uniform
     * @param array<int> $shape
     */
    public function randomUniform(
        array $shape,
        int|float $low,
        int|float $high,
        ?int $dtype=null,
        ?int $seed=null,
        ?NDArray $X=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomUniform");
        }
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->alloc($shape,dtype:$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
        }
        if($seed===null||$seed===0) {
            //$rngState = $this->rngState;
            $seed = $this->scalar($this->randInt64());
        } else {
            //$rngState = $this->createRngState($seed,self::DEFAULT_PCG32_SEQUENCE);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        //$this->openclmath->pcg32Uniform(
        //    $rngState,
        //    $n,
        //    $XX,$offX,1,
        //    $low,
        //    $high,
        //    $events, $waitEvents
        //);
        //$this->openclmath->pcg32Step(
        //    $rngState,
        //    $events, $waitEvents
        //);
        $this->openclmath->splitmix64Uniform(
            $n,
            $XX,$offX,1,
            $low,
            $high,
            $seed,
            $events, $waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomUniform");
        }
        return $X;
    }

    /**
     * @param array<int> $shape
     */
    public function randomNormal(
        array $shape,
        float $mean,
        float $scale,
        ?int $dtype=null,
        ?int $seed=null,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomNormal");
        }
        if($dtype!==null&&$output!==null) {
            if ($output->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of output');
            }
        }
        if($output===null) {
            $output = $this->alloc($shape,dtype:$dtype);
        } else {
            if ($output->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of output');
            }
        }
        if($seed===null||$seed===0) {
            $seed = $this->scalar($this->randInt64());
        }

        $n = $output->size();
        $XX = $output->buffer();
        $offX = $output->offset();

        //$this->openclmath->pcg32Normal(
        //    $this->rngState,
        //    $n,
        //    $XX,$offX,1,
        //    $mean,
        //    $scale,
        //    $events, $waitEvents
        //);
        //$this->openclmath->pcg32Step(
        //    $this->rngState,
        //    $events, $waitEvents
        //);
        $this->openclmath->splitmix64Normal(
            $n,
            $XX,$offX,1,
            $mean,
            $scale,
            $seed,
            $events, $waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomNormal");
        }
        return $output;
    }

    public function randomSequence(
        int $base,
        ?int $size=null,
        ?int $seed=null,
        ?int $dtype=null,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomSequence");
        }
        if($size==null) {
            $size = $base;
        }
        if($output==null) {
            $dtype = $dtype ?? NDArray::int32;
            $output = $this->alloc([$size],dtype:$dtype);
        } else {
            $dtype = $dtype ?? $output->dtype();
            if($output->dtype()!=$dtype || $output->size()!=$size) {
                throw new InvalidArgumentException("output size must be the same of size and same dtype");
            }
        }
        if($seed===null) {
            $seed = $this->scalar($this->randInt64());
        }

        $n = $base;
        $XX = $output->buffer();
        $offX = $output->offset();

        $this->openclmath->randomSequence(
            $n,
            $size,
            $XX,$offX,1,
            $seed,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomSequence");
        }
        return $output;
    }

    /**
     * $probs : (batches,numClasses) dtype:float32.
     * $randints: (batches) dtype:int32
     * 
     * sum of probs must be 1.0 each row.
     */
    public function randomCategorical(
        NDArray $logits,
        ?int $numSamples=null,
        ?int $dtype=null,
        ?int $seed=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomCategorical");
        }
        $la = $this;
        if(!$la->isFloat($logits)) {
            throw new InvalidArgumentException('logits must be float dtype. '.$this->dtypeToString($logits->dtype()).' given.');
        }
        if($numSamples===null) {
            if($logits->ndim()!=2) {
                throw new InvalidArgumentException('logits must be 2D NDArray. '.$this->shapeToString($logits->shape()).' given.');
            }
        } else {
            if($numSamples<1) {
                throw new InvalidArgumentException('numSamples must be positive.');
            }
            if($logits->ndim()!=1) {
                throw new InvalidArgumentException('logits must be 1D NDArray. '.$this->shapeToString($logits->shape()).' given.');
            }
        }
        if($dtype===null) {
            $dtype = NDArray::int32;
        }
        //if($numSamples===null) {
        //    if($probs->ndim()!=2) {
        //        throw new InvalidArgumentException('probs must be 2D NDArray without numSamples.');
        //    }
        //    [$batches,$numClasses] = $probs->shape();
        //    $waitPrev = $waitEvents;
        //    $waitEvents = $this->newEventList();
        //    $rand = $la->randomUniform(
        //        [$batches],dtype:$probs->dtype(),low:0.0,high:1.0,seed:$seed,
        //        events:$waitEvents,waitEvents:$waitPrev
        //    );// (batches)
        //    $waitPrev = $waitEvents;
        //    $waitEvents = $this->newEventList();
        //    $thresholds = $la->cumsum(
        //        $probs,axis:-1,
        //        events:$waitEvents,waitEvents:$waitPrev
        //    );      // (batches,numClasses)
        //    $waitPrev = $waitEvents;
        //    $waitEvents = $this->newEventList();
        //    $randints = $la->searchsorted(                          // (batches)
        //        $thresholds,$rand,
        //        right:true,dtype:$dtype,
        //        events:$events,waitEvents:$waitPrev
        //    );
        //} else {
        //    if($probs->ndim()!=1) {
        //        throw new InvalidArgumentException('probs must be 1D NDArray with numSamples.');
        //    }
        //    $numClasses = $probs->shape()[0];
        //    $batches = $numSamples;
        //    $waitPrev = $waitEvents;
        //    $waitEvents = $this->newEventList();
        //    $rand = $la->randomUniform(
        //        [$batches],dtype:$probs->dtype(),low:0.0,high:1.0,seed:$seed,
        //        events:$waitEvents,waitEvents:$waitPrev
        //    );// (batches)
        //    $waitPrev = $waitEvents;
        //    $waitEvents = $this->newEventList();
        //    $thresholds = $la->cumsum(
        //        $probs,
        //        events:$waitEvents,waitEvents:$waitPrev
        //    );      // (numClasses)
        //    $waitPrev = $waitEvents;
        //    $waitEvents = $this->newEventList();
        //    $randints = $la->searchsorted(          // (batches)
        //        $thresholds,    // (numClasses) :  NON individual mode
        //        $rand,          // (batches)
        //        right:true,
        //        dtype:$dtype,
        //        events:$events,waitEvents:$waitPrev
        //    );
        //    
        //}
        if($numSamples===null) {
            $numSamples = 1;
            [$batches,$numClasses] = $logits->shape();
            $randints = $la->alloc([$batches],dtype:$dtype);
        } else {
            $batches = 1;
            [$numClasses] = $logits->shape();
            $randints = $la->alloc([$numSamples],dtype:$dtype);
        }
        if($seed===null || $seed===0) {
            $seed = $la->scalar($la->randInt64());
        }
        $this->openclmath->splitmix64Categorical(
            $batches,
            $numClasses,
            $numSamples,
            $logits->buffer(), $logits->offset(),
            $randints->buffer(), $randints->offset(),
            $seed,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomCategorical");
        }
        return $randints;
    }

    /**
     * Returns the number of occurrences of each value in an array.
     */
    public function bincount(
        NDArray $input,
        int $numClass,
        ?bool $trans=null,
        ?bool $addMode=null,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("bincount");
        }
        $trans ??= false;
        $addMode ??= false;
        $dtype = $input->dtype();
        if($dtype!==NDArray::int32 && $dtype!==NDArray::int64) {
            throw new InvalidArgumentException('input must be int32 or int64 dtype.');
        }
        $ndim = $input->ndim();
        if($ndim!=1&&$ndim!=2) {
            throw new InvalidArgumentException('input must be 1D or 2D NDArray.');
        }
        if($numClass<1) {
            throw new InvalidArgumentException('numClass must be positive.');
        }

        if($ndim==1) {
            $m = 1;
            $n = $input->shape()[0];
            $outputShape = [$numClass];
        } else {
            [$m,$n] = $input->shape();
            if($trans) {
                $outputShape = [$numClass,$n];
            } else {
                $outputShape = [$m,$numClass];
            }
        }

        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                throw new InvalidArgumentException('output shape must match expected shape.');
            }
            if($output->dtype()!=$dtype) {
                throw new InvalidArgumentException('output dtype must match expected dtype.');
            }
        }
        $this->openclmath->bincount(
            $trans,
            $m,
            $n,
            $numClass,
            $addMode,
            $input->buffer(), $input->offset(),
            $output->buffer(), $output->offset(),
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("bincount");
        }
        return $output;
    }

    /**
     *  Y = X(n)
     * @param array<int> $begin
     * @param array<int> $size
     */
    public function slice(
        NDArray $input,
        array $begin,
        array $size,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("slice");
        }
        $output = $this->doSlice(
            false,
            $input,
            $begin,
            $size,
            $output,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("slice");
        }
        return $output;
    }

    /**
     * @param array<int> $begin
     * @param array<int> $size
     */
    public function stick(
        NDArray $input,
        NDArray $output,
        array $begin,
        array $size,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("stick");
        }
        $output = $this->doSlice(
            true,
            $output,
            $begin,
            $size,
            $input,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("stick");
        }
        return $output;
    }

    /**
     * @param array<int> $begin
     * @param array<int> $size
     */
    protected function doSlice(
        bool $reverse,
        NDArray $input,
        array $begin,
        array $size,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("doSlice");
        }
        if(!$reverse){
            $messageInput='Input';
        } else {
            $messageInput='Output';
        }
        $orgBegin = $begin;
        $orgSize = $size;
        $ndimBegin = count($begin);
        if($ndimBegin<1||$ndimBegin>3) {
            throw new InvalidArgumentException('begin must has 1 or 2 or 3 integer.');
        }
        $ndimSize = count($size);
        if($ndimSize<1||$ndimSize>3) {
            throw new InvalidArgumentException('Size must has 1 or 2 or 3 integer.');
        }
        if($ndimBegin!=$ndimSize){
            throw new InvalidArgumentException('Unmatch shape of begin and size');
        }
        $ndimInput = $input->ndim();
        if($ndimInput<$ndimBegin){
            throw new InvalidArgumentException($messageInput.' shape rank is low to slice');
        }
        $shape = $input->shape();

        // ndim = 0
        $m = array_shift($shape);
        $startAxis0 = array_shift($begin);
        if($startAxis0<0){
            $startAxis0 = $m+$startAxis0;
        }
        if($startAxis0<0||$startAxis0>=$m){
            throw new InvalidArgumentException('start of axis 0 is invalid value.');
        }
        $sizeAxis0 = array_shift($size);
        if($sizeAxis0<0){
            $sizeAxis0 = $m-$startAxis0+$sizeAxis0+1;
        }
        if($sizeAxis0<1||$startAxis0+$sizeAxis0>$m){
            throw new InvalidArgumentException('size of axis 0 is invalid value.');
        }

        // ndim = 1
        if($ndimBegin<=1){
            $n = 1;
            $startAxis1 = 0;
            $sizeAxis1 = 1;
        } else {
            $n = array_shift($shape);
            $startAxis1 = array_shift($begin);
            if($startAxis1<0){
                $startAxis1 = $n+$startAxis1;
            }
            if($startAxis1<0||$startAxis1>=$n){
                throw new InvalidArgumentException('start of axis 1 is invalid value.:begin=['.implode(',',$orgBegin).']');
            }
            $sizeAxis1 = array_shift($size);
            if($sizeAxis1<0){
                $sizeAxis1 = $n-$startAxis1+$sizeAxis1+1;
            }
            if($sizeAxis1<1||$startAxis1+$sizeAxis1>$n){
                throw new InvalidArgumentException('size of axis 1 is invalid value.');
            }
        }

        // ndim = 2
        if($ndimBegin<=2){
            $k = 1;
            $startAxis2 = 0;
            $sizeAxis2 = 1;
        } else {
            $k = array_shift($shape);
            $startAxis2 = array_shift($begin);
            if($startAxis2<0){
                $startAxis2 = $k+$startAxis2;
            }
            if($startAxis2<0||$startAxis2>=$k){
                throw new InvalidArgumentException('start of axis 2 is invalid value.:begin=['.implode(',',$orgBegin).']');
            }
            $sizeAxis2 = array_shift($size);
            if($sizeAxis2<0){
                $sizeAxis2 = $k-$startAxis2+$sizeAxis2+1;
            }
            if($sizeAxis2<1||$startAxis2+$sizeAxis2>$k){
                throw new InvalidArgumentException('size of axis 2 is invalid value.');
            }
        }
        $itemSize = array_product($shape);
        $outputShape = [$sizeAxis0];
        if($ndimBegin>=2){
            array_push($outputShape,
                $sizeAxis1);
        }
        if($ndimBegin>=3){
            array_push($outputShape,
                $sizeAxis2);
        }
        $outputShape = array_merge(
            $outputShape,$shape);
        if($output==null){
            $output = $this->alloc($outputShape,dtype:$input->dtype());
        }else{
            if($outputShape!=$output->shape()){
                throw new InvalidArgumentException('Unmatch output shape: '.
                    $this->printableShapes($output->shape()).'<=>'.
                    $this->printableShapes($outputShape));
            }
        }

        $A = $input->buffer();
        $offsetA = $input->offset();
        $Y = $output->buffer();
        $offsetY = $output->offset();
        $incA = 1;
        $incY = 1;
        $this->openclmath->slice(
            $reverse,
            $addMode=false,
            $m,
            $n,
            $k,
            $itemSize,
            $A,$offsetA,$incA,
            $Y,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2,
            $events,$waitEvents
        );
        if($this->profiling) {
            $this->profilingEnd("doSlice");
        }
        return $output;
    }

    /**
     * repeat
     */
    public function repeat(
        NDArray $A,
        int $repeats,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("repeat");
        }
        if($repeats<1) {
            throw new InvalidArgumentException('repeats argument must be one or greater.');
        }
        if($axis!==null) {
            $ndim = $A->ndim();
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            if($A->ndim()<$axis) {
                throw new InvalidArgumentException('dimension rank must be two or greater.');
            }
        }
        $innerShape = $A->shape();
        $outerShape = [];
        if($axis!==null) {
            //for($i=0;$i<$axis;$i++) {
            //    $outerShape[] = array_shift($innerShape);
            //}
            $outerShape = array_splice($innerShape,0,$axis);
        }
        $base = 1;
        if($axis===null) {
            $outputShape = [(int)array_product(
                    array_merge($outerShape,[$repeats],$innerShape))];
        } else {
            if($keepdims) {
                $base = array_shift($innerShape);
                if($base===null) {
                    throw new InvalidArgumentException('dimension rank must be two or greater on keepdims.');
                }
                $outputShape = array_merge($outerShape,[$repeats*$base],$innerShape);
            } else {
                $outputShape = array_merge($outerShape,[$repeats],$innerShape);
            }
        }
        $B = $this->alloc($outputShape,dtype:$A->dtype());
        $m = (int)array_product($outerShape);
        $k = (int)array_product($innerShape)*$base;
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->openclmath->repeat(
            $m,
            $k,
            $repeats,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("repeat");
        }
        return $B;
    }

    /**
     * @param array<int>|NDArray $perm
     */
    public function transpose(
        NDArray $A,
        array|NDArray|null $perm=null,
        ?NDArray $B=null,
        ?object $events=null
        ) : NDArray
    {
        if($A->ndim()<1) {
            throw new InvalidArgumentException('input array must be grator than or equal 1D.');
        }
        if($A->ndim()==2 && $this->isFloat($A)) {
            if($perm) {
                if(count($perm)!=2) {
                    throw new InvalidArgumentException('unmatch sourceshape and perm');
                    //if(!is_array($perm)) {
                    //    $perm = $perm->toArray();
                    //}
                    //if($perm==[0,1]) {
                    //    return $this->copy($A,$B);
                    //}
                    //if($perm!=[1,0]) {
                    //    throw new InvalidArgumentException('unmatch sourceshape and perm');
                    //}
                }
            }
            return $this->transpose2D($A,$B,$events);
        }
        return $this->transposeND($A,$perm,$B,$events);
    }

    /**
     * @param array<int>|NDArray $perm
     */
    protected function transposeND(
        NDArray $A,
        array|NDArray|null $perm=null,
        ?NDArray $B=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("transposeND");
        }
        if($perm===null) {
            $perm = range($A->ndim()-1,0,-1);
            $perm = new NDArrayPhp($perm,dtype:NDArray::int32,service:$this->service);
        } elseif(is_array($perm)) {
            $perm = new NDArrayPhp($perm,dtype:NDArray::int32,service:$this->service);
        }
        $perm = $this->toNDArray($perm);
        $shapeA = $A->shape();
        if(count($shapeA)!=count($perm)) {
            throw new InvalidArgumentException('unmatch sourceshape and perm');
        }
        $shapeB = [];
        $checkPerm = [];
        foreach($perm->toArray() as $axis) {
            if(isset($checkPerm[$axis])) {
                throw new InvalidArgumentException('duplicate axis in perm option');
            }
            $checkPerm[$axis] = true;
            $shapeB[] = $shapeA[$axis];
        }
        if($B===null) {
            $B = $this->alloc($shapeB,dtype:$A->dtype());
        } else {
            if($B->shape()!=$shapeB) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }

        //$AA = $this->toNDArray($A)->buffer();
        //$BB = $this->newHostBuffer($B->size(),$B->dtype());
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offsetA = $A->offset();
        //$offsetB = 0;
        $offsetB = $B->offset();
        $sourceShape = (new NDArrayPhp($shapeA,dtype:NDArray::int32,service:$this->service))->buffer();
        $permBuf = $perm->buffer();
        $this->openclmath->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB, 
            $events,$waitEvents
        );
        //$B->buffer()->write(
        //    $this->queue,
        //    $BB,                            // host buffer
        //    $BB->valueSize()*$B->size(),   // bytes
        //    $BB->valueSize()*$B->offset(), // offset bytes
        //);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("transposeND");
        }
        return $B;
    }

    public function transpose2D(
        NDArray $A,
        ?NDArray $B=null,
        ?object $events=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("transpose2D");
        }
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('input array must be 2D.');
        }
        $shape = $A->shape();
        $shape = [$shape[1],$shape[0]];
        if($B==null) {
            $B = $this->alloc($shape,dtype:$A->dtype());
        }/* else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }*/
        $m = $shape[1];
        $n = $shape[0];
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->blas->omatcopy(
            BLAS::RowMajor,
            BLAS::Trans,
            $m,
            $n,
            $alpha=1.0,
            $AA,$offA,$n,
            $BB,$offB,$m,
            $this->queue,$events
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("transpose2D");
        }
        return $B;
    }

    public function bandpart(
        NDArray $A,
        int $lower,
        int $upper,
        ?object $events=null, ?object $waitEvents=null
    ) : void
    {
        if($this->profiling) {
            $this->profilingStart("bandpart");
        }
        if($A->ndim()<2) {
            throw new InvalidArgumentException('input array must be 2D or upper.');
        }
        $shape = $A->shape();
        $k = array_pop($shape);
        $n = array_pop($shape);
        $m = (int)array_product($shape);
        $buffer = $A->buffer();
        $offset = $A->offset();
        $this->openclmath->bandpart(
            $m,$n,$k,
            $buffer,$offset,
            $lower,
            $upper,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("bandpart");
        }
    }

    public function imagecopy(
        NDArray $A,
        ?NDArray $B=null,
        ?bool $channels_first=null,
        ?int $heightShift=null,
        ?int $widthShift=null,
        ?bool $verticalFlip=null,
        ?bool $horizontalFlip=null,
        ?bool $rgbFlip=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("imagecopy");
        }
        if($A->ndim()!=3) {
            throw new InvalidArgumentException('input array must be 3D.');
        }
        $shape = $A->shape();
        if($B==null) {
            $B = $this->alloc($shape,dtype:$A->dtype());
            //$zerosEvents = $this->newEventList();
            $this->zeros($B,/*events:$zerosEvents*/);
            //$zerosEvents->wait();
        } else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }
        if($heightShift==null) {
            $heightShift=0;
        }
        if($widthShift==null) {
            $widthShift=0;
        }
        if($verticalFlip==null) {
            $verticalFlip=false;
        }
        if($horizontalFlip==null) {
            $horizontalFlip=false;
        }
        if($rgbFlip==null) {
            $rgbFlip=false;
        }
        if($channels_first==null) {
            $channels_first=false;
            $height = $shape[0];
            $width = $shape[1];
            $channels = $shape[2];
        } else {
            $channels_first=true;
            $channels = $shape[0];
            $height = $shape[1];
            $width = $shape[2];
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->openclmath->imagecopy(
            $height,
            $width,
            $channels,
            $AA, $offA,
            $BB, $offB,
            $channels_first,
            $heightShift,
            $widthShift,
            $verticalFlip,
            $horizontalFlip,
            $rgbFlip,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("imagecopy");
        }
        return $B;
    }

    public function abs(
        float|int|object $value,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : float|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("abs");
        }
        if($value instanceof NDArray) {
            if($this->isComplexDtype($value->dtype())) {
                $ftype = ($value->dtype()==NDArray::complex64) ? NDArray::float32 : NDArray::float64;
                if($output==null) {
                    $output = $this->alloc($value->shape(),dtype:$ftype);
                } else {
                    if($output->dtype()!=$ftype) {
                        throw new InvalidArgumentException('output dtype must be same as input.');
                    }
                }
                $this->openclmath->absComplex(
                    $value->size(),
                    $value->buffer(),$value->offset(),1,
                    $output->buffer(),$output->offset(),1,
                    $events,$waitEvents
                );
            } elseif($this->isFloat($value)) {
                if($output==null) {
                    $output = $this->copy($value,events:$events,waitEvents:$waitEvents);
                } else {
                    if($output->dtype()!=$value->dtype()) {
                        throw new InvalidArgumentException('output dtype must be same as input.');
                    }
                    $this->copy($value,$output,events:$events,waitEvents:$waitEvents);
                }
                $this->openclmath->abs(
                    $output->size(),
                    $output->buffer(),$output->offset(),1,
                    $events,$waitEvents
                );
            } else {
                $dtypeString = $this->dtypeToString($value->dtype());
                throw new InvalidArgumentException("Unsupported dtype: $dtypeString");
            }
        } elseif(is_numeric($value)) {
            if($waitEvents) {
                $waitEvents->wait();
            }
            $output = abs($value);
            if($events) {
                $events->move($this->newEventList());
            }

        } elseif($this->isComplexObject($value)) {
            if($waitEvents) {
                $waitEvents->wait();
            }
            $output = $this->cabs($value);
            if($events) {
                $events->move($this->newEventList());
            }
        } else {
            throw new InvalidArgumentException("invalid data type: ".$this->dataTypeString($value));
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("abs");
        }
        return $output;
    }

    //protected function absNDArray(
    //    NDArray $value,
    //    ?object $events=null, ?object $waitEvents=null
    //    ) : NDArray
    //{
    //    if($this->isFloat($value)) {
    //        if($this->profiling) {
    //            $this->profilingStart("absNDArray");
    //        }
    //        $waitPrev = $waitEvents;
    //        $waitEvents = $this->newEventList();
    //        $conditionValue = $this->copy($value,events:$waitEvents,waitEvents:$waitPrev);
    //        $waitPrev = $waitEvents;
    //        $waitEvents = $this->newEventList();
    //        $this->greaterEqual($conditionValue,0,events:$waitEvents,waitEvents:$waitPrev);
    //        $waitPrev = $waitEvents;
    //        $waitEvents = $this->newEventList();
    //        $this->scal(2,$conditionValue,events:$waitEvents,waitEvents:$waitPrev);
    //        $waitPrev = $waitEvents;
    //        $waitEvents = $this->newEventList();
    //        $this->increment($conditionValue,beta:-1,events:$waitEvents,waitEvents:$waitPrev);
    //        $waitPrev = $waitEvents;
    //        $this->multiply($conditionValue,$value,events:$events,waitEvents:$waitPrev);
    //        if($this->blocking) {
    //            $this->finish();
    //        }
    //        if($this->profiling) {
    //            $this->profilingEnd("absNDArray");
    //        }
    //        return $value;
    //    } else {
    //        $dtypeString = $this->dtypeToString($value->dtype());
    //        throw new InvalidArgumentException("Unsupported dtype: $dtypeString");
    //    }
    //}

    //public function where(
    //    NDArray $condition,
    //    NDArray $x,
    //    NDArray $y,
    //    ?bool $normalize=null,
    //    ?object $events=null, ?object $waitEvents=null
    //    ) : NDArray
    //{
    //    if($this->profiling) {
    //        $this->profilingStart("where");
    //    }
    //    $normalize ??= true;
    //    $origShape = $x->shape();
    //    $condition = $condition->reshape([$condition->size()]); // (size)
    //    $x = $x->reshape([$x->size()]); // (size)
    //    $y = $y->reshape([$y->size()]); // (size)
//
    //    $waitPrev = $waitEvents;
    //    $waitEvents = $this->newEventList();
    //    $xy = $this->stack([$y,$x],events:$waitEvents,waitEvents:$waitPrev);        // (2,size)
//
    //    if($normalize) {
    //        if($condition->dtype()!==NDArray::bool) {
    //            $waitPrev = $waitEvents;
    //            $waitEvents = $this->newEventList();
    //            $condition = $this->astype($condition,dtype:NDArray::bool,events:$waitEvents,waitEvents:$waitPrev);
    //        }
    //    }
//
    //    if($condition->dtype()!=NDArray::int32) {
    //        $waitPrev = $waitEvents;
    //        $waitEvents = $this->newEventList();
    //        $condition = $this->astype($condition,dtype:NDArray::int32,events:$waitEvents,waitEvents:$waitPrev);
    //    }
//
    //    $waitPrev = $waitEvents;
    //    $waitEvents = $this->newEventList();
    //    $result = $this->gatherb($xy,$condition,detailDepth:2,indexDepth:0,events:$waitEvents,waitEvents:$waitPrev); // (size)
//
    //    $result = $result->reshape($origShape);
    //    if($this->blocking) {
    //        $this->finish();
    //    }
    //    if($this->profiling) {
    //        $this->profilingEnd("where");
    //    }
    //    return $result;
    //}

    /**
     *    Out(m,n) := Cond(m,n) ? X(m,n) : Y(m,n)
     *
     *    The operand with the largest number of dimensions among Cond, X, and Y becomes the output shape (base shape).
     *    Operands with fewer dimensions are treated as vectors matching the trailing (trans=false) or
     *    leading (trans=true) dimensions of the base shape.
     *    Numbers (scalars) can also be specified for X and Y.
     *
     *    Operands with the same number of dimensions as the base shape must have the identical shape.
     *    If there are multiple operands treated as vectors, they must have the same shape.
     */
    
    public function where(
        NDArray $condition,
        NDArray|float|int $X,
        NDArray|float|int $Y,
        ?bool $trans=null,
        ?NDArray $output=null,
        ?object $events=null, ?object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("where");
        }
        $trans ??= false;

        if($condition->dtype()!=NDArray::bool) {
            throw new InvalidArgumentException('condition must be bool.');
        }

        // Determine the dtype of X and Y (if both are NDArrays, they must match)
        $dtype = null;
        foreach([$X,$Y] as $v) {
            if($v instanceof NDArray) {
                if($dtype!==null && $dtype!=$v->dtype()) {
                    throw new InvalidArgumentException('Unmatch dtype of X and Y.');
                }
                $dtype = $v->dtype();
            }
        }
        $dtype ??= NDArray::float32;

        // Transform args to ndarray and validate broadcast shape
        [$ops,$baseShape,$kinds,$m,$n,$total] = $this->transformBroadcastShape([
            $condition,
            $X,
            $Y,
        ],dtype:$dtype,trans:$trans);

        if($output===null) {
            $output = $this->alloc($baseShape,dtype:$dtype);
        } else {
            if($output->shape()!=$baseShape) {
                throw new InvalidArgumentException('output shape does not match base shape.');
            }
            if($output->dtype()!=$dtype) {
                throw new InvalidArgumentException('output dtype does not match base dtype.');
            }
        }
        if($total>0) {
            [[$cc,],[$xx,],[$yy,]] = $ops;
            //echo "=====================\n";
            //echo "trans:".var_export($trans,true)."\n";
            //echo "m:".$m.",n:".$n."\n";
            //echo "C->count:".$cc->buffer()->count().", offsetC: ".$cc->offset().", kindC:".$kinds[0]."\n";
            //echo "X->count:".$xx->buffer()->count().", offsetX: ".$xx->offset().", kindX:".$kinds[1]."\n";
            //echo "Y->count:".$yy->buffer()->count().", offsetY: ".$yy->offset().", kindY:".$kinds[2]."\n";
            //echo "out->count:".$out->buffer()->count().", offsetOut: ".$out->offset()."\n";
            //throw new \RuntimeException("Test");
            $this->openclmath->where(
                $trans,
                $m,
                $n,
                $cc->buffer(),$cc->offset(),$kinds[0],
                $xx->buffer(),$xx->offset(),$kinds[1],
                $yy->buffer(),$yy->offset(),$kinds[2],
                $output->buffer(),$output->offset(),
                events:$events, waitEvents:$waitEvents
            );
        }

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("where");
        }
        return $output;
    }

    /**
     * @return array{array<int>,array<array<int>>,array<int>}
     */
    public function parseEinsum(
        string $equation,
        NDArray ...$arrays,
    ) : array
    {
        [$a,$b] = $arrays;
        $equation = str_replace(' ','',$equation);
        if(!preg_match(
            "/([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)/",
            $equation,
            $split_string
        )) {
            throw new InvalidArgumentException("Not recognized as an equation: '{$equation}'");
        }
        $labelA = $split_string[1];
        $labelB = $split_string[2];
        $labelC = $split_string[3];
        if($a->ndim()!=strlen($labelA)) {
            $msg = "(".implode(',',$a->shape()).")";
            throw new InvalidArgumentException("The rank of array A does not match the equation: '{$equation}', but {$msg} given.");
        }
        if($b->ndim()!=strlen($labelB)) {
            $msg = "(".implode(',',$b->shape()).")";
            throw new InvalidArgumentException("The rank of array B does not match the equation: '{$equation}', but {$msg} given.");
        }

        $allIndices = [];
        $labels = [$labelA,$labelB];
        $shapes = [$a->shape(),$b->shape()];
        foreach($labels as $idx => $label) {
            foreach(str_split($label) as $axis => $chr) {
                if(array_key_exists($chr,$allIndices)) {
                    if($allIndices[$chr]!=$shapes[$idx][$axis]) {
                        $shapeError = '('.implode(',',$a->shape()).'),('.implode(',',$b->shape()).')';
                        throw new InvalidArgumentException("Unmatch shapes in index letter '{$chr}' in equation:'{$equation}', but shapes {$shapeError} given.");
                    }
                } else {
                    $allIndices[$chr] = $shapes[$idx][$axis];
                }
            }
        }
        $outputShape = [];
        foreach(str_split($labelC) as $axis => $chr) {
            if(!array_key_exists($chr,$allIndices)) {
                throw new InvalidArgumentException("Target index letter '{$chr}' not found in inputs.:'{$equation}'");
            }
            $outputShape[$chr] = $allIndices[$chr];
            unset($allIndices[$chr]);
        }
        $allIndices = array_merge($outputShape,$allIndices);
        $outputShape = array_values($outputShape);

        // complie labels
        $allLabels = array_keys($allIndices);
        $sizeOfIndices = array_values($allIndices);
        $labels = [$labelA,$labelB,$labelC];
        $compiledLabels = [];
        foreach($labels as $label) {
            $compiledLabel = [];
            foreach(str_split($label) as $axis => $chr) {
                $compiledLabel[] = array_search($chr,$allLabels);
            }
            $compiledLabels[] = $compiledLabel;
        }

        return [
            $sizeOfIndices,
            $compiledLabels,
            $outputShape,
        ];
    }

    /**
     * @param array<int> $shapeA
     * @param array<int> $labelA
     * @param array<int> $shapeB
     * @param array<int> $labelB
     * @param array<int> $labelC
     * @return array{0:array<int>,1:array<int>}
     */
    public function einsum_build_dims(
        string $equation,
        array $shapeA,
        array $labelA,
        array $shapeB,
        array $labelB,
        array $labelC,
    ) : array
    {
        if(count($shapeA)!=count($labelA)) {
            throw new InvalidArgumentException(
                "Invalid rank of arg A.: (".implode(',',$shapeA)."), '{$equation}' given."
            );
        }
        if(count($shapeB)!=count($labelB)) {
            throw new InvalidArgumentException(
                "Invalid rank of arg B.: (".implode(',',$shapeB)."), '{$equation}' given."
            );
        }
        $sizeOfIndices = [];
        foreach(array_map(null,$shapeA,$labelA) as [$dim,$lbl]) {
            if(array_key_exists($lbl,$sizeOfIndices)) {
                if($sizeOfIndices[$lbl] != $dim) {
                    throw new InvalidArgumentException(
                        "Invalid shape of args: a(".implode(',',$shapeA)."),b(".implode(',',$shapeB)."), '{$equation}' given."
                    );
                }
            } else {
                $sizeOfIndices[$lbl] = $dim;
            }
        }
        foreach(array_map(null,$shapeB,$labelB) as [$dim,$lbl]) {
            if(array_key_exists($lbl,$sizeOfIndices)) {
                if($sizeOfIndices[$lbl] != $dim) {
                    throw new InvalidArgumentException(
                        "Invalid shape of args: a(".implode(',',$shapeA)."),b(".implode(',',$shapeB)."), '{$equation}' given."
                    );
                }
            } else {
                $sizeOfIndices[$lbl] = $dim;
            }
        }
        ksort($sizeOfIndices);
        $outputShape = array_slice($sizeOfIndices,0,count($labelC));
        return [$sizeOfIndices,$outputShape];
    }

    /**
     * @param array<int> $sizeOfIndices
     * @param array<int> $label
     * @param array<int> $shape
     * @return array<int>
     */
    public function einsum_build_lds(
        array $sizeOfIndices,
        array $label,
        array $shape,
    ) : array
    {
        $lds = array_fill(0,count($sizeOfIndices),0);
        $ld = 1;
        foreach(array_reverse($label,true) as $axis => $lbl) {
            $broadcast = false;
            if($shape[$axis]==1) {
                $broadcast = true;
            }
            if(!$broadcast) {
                $lds[$lbl] += $ld;
            }
            $ld *= $sizeOfIndices[$lbl];
        }
        return $lds;
    }

    public function einsum(
        string $equation,
        NDArray $a,
        NDArray $b,
        ?object $events=null,
        ?object $waitEvents=null 
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("einsum");
        }
        $shapeA = $a->shape();
        $shapeB = $b->shape();
        $cacheKey = $equation;
        if(isset($this->einsumEquationCache[$cacheKey])) {
            [
                $compiledLabels,
            ] = $this->einsumEquationCache[$cacheKey];
            [$labelA,$labelB,$labelC] = $compiledLabels;
            [
                $sizeOfIndices,
                $outputShape,
            ] = $this->einsum_build_dims($equation,$shapeA,$labelA,$shapeB,$labelB,$labelC);
        } else {
            [
                $sizeOfIndices,
                $compiledLabels,
                $outputShape,
            ] = $this->parseEinsum($equation, $a, $b);
            [$labelA,$labelB,$labelC] = $compiledLabels;
        }
        $ldA = $this->einsum_build_lds($sizeOfIndices,$labelA,$shapeA);
        $ldB = $this->einsum_build_lds($sizeOfIndices,$labelB,$shapeB);
        $sizeOfIndices = $this->array($sizeOfIndices,dtype:NDArray::int32)->buffer();
        $ldA = $this->array($ldA,dtype:NDArray::int32)->buffer();
        $ldB = $this->array($ldB,dtype:NDArray::int32)->buffer();

        $outputs = $this->alloc($outputShape,dtype:$a->dtype());
        //$this->zeros($outputs);

        $AA = $a->buffer();
        $offsetA = $a->offset();
        $BB = $b->buffer();
        $offsetB = $b->offset();
        $CC = $outputs->buffer();
        $offsetC = $outputs->offset();
        $ndimC = $outputs->ndim();

        $this->openclmath->einsum(
            $sizeOfIndices,
            $AA,
            $offsetA,
            $ldA,
            $BB,
            $offsetB,
            $ldB,
            $CC,
            $offsetC,
            $ndimC,
            events:$events,
            waitEvents:$waitEvents,
        );

        if($this->blocking) {
            $this->finish();
        }
        if(!isset($this->einsumEquationCache[$cacheKey])) {
            $this->einsumEquationCache[$cacheKey] = 
            [
                $compiledLabels,
            ];
        }
        if($this->profiling) {
            $this->profilingEnd("einsum");
        }
        return $outputs;
    }

    public function einsum4p1(
        string $equation,
        NDArray $a,
        NDArray $b,
        ?NDArray $c=null,
        ?object $events=null,
        ?object $waitEvents=null 
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("einsum4p1");
        }

        $shapeA = $a->shape();
        $shapeB = $b->shape();
        if(isset($this->einsum4p1EquationCache[$equation])) {
            [
                $compiledLabels,
            ] = $this->einsum4p1EquationCache[$equation];
            [$labelA,$labelB,$labelC] = $compiledLabels;
            [
                $dims,
                $shapeC,
            ] = $this->einsum_build_dims($equation,$shapeA,$labelA,$shapeB,$labelB,$labelC);
            $sumDims = count($dims)-count($labelC);
            $inputMaxDims = 5-(1-$sumDims);
        } else {
            [
                $dims,
                $compiledLabels,
                $shapeC,
            ] = $this->parseEinsum($equation, $a, $b);
            [$labelA,$labelB,$labelC] = $compiledLabels;
            $sumDims = count($dims)-count($labelC);
            $inputMaxDims = 5-(1-$sumDims);
            if(count($labelC)>4) {
                throw new InvalidArgumentException("rank of C must be less than 4D or equal. (".implode(',',array_slice($dims,0,-count($labelC))).") given.");
            }
            if($sumDims>1) {
                throw new InvalidArgumentException("rank of overlay dims must be less than 1D or equal. '{$equation}' and {$sumDims} sum dims given.");
            }
            if(count($dims)>$inputMaxDims) {
                throw new InvalidArgumentException("dims must be less than {$inputMaxDims}D or equal.");
            }
            if(count($labelA)>$inputMaxDims) {
                throw new InvalidArgumentException("rank of A must be less than {$inputMaxDims}D or equal. (".implode(',',$a->shape()).") given.");
            }
            if(count($labelB)>$inputMaxDims) {
                throw new InvalidArgumentException("rank of B must be less than {$inputMaxDims}D or equal. (".implode(',',$b->shape()).") given.");
            }
        }
        $ldA = $this->einsum_build_lds($dims,$labelA,$shapeA);
        $ldB = $this->einsum_build_lds($dims,$labelB,$shapeB);

        if($c==null) {
            $c = $this->alloc($shapeC,dtype:$a->dtype());
        } else {
            if($c->shape()!=$shapeC) {
                throw new InvalidArgumentException(
                    "unmatch shape of C. the shape must be (".implode(',',$shapeC)."). ".
                    "(".implode(',',$c->shape()).") given.");
            }
        }
        if($sumDims==0) {
            $dims[] = 1;
            $ldA[]  = 1;
            $ldB[]  = 1;
        }
        //echo "equation=$equation\n";
        //echo "labelA=(".implode(',',$labelA).")\n";
        //echo "labelB=(".implode(',',$labelB).")\n";
        //echo "labelC=(".implode(',',$labelC).")\n";
        //echo "shapeC=(".implode(',',$shapeC).")\n";
        //echo "dims=(".implode(',',$dims).")\n";
        //echo "ldA=(".implode(',',$ldA).")\n";
        //echo "ldB=(".implode(',',$ldB).")\n";
        $dims = array_pad($dims,-5,1);
        $ldA = array_pad($ldA,-5,0);
        $ldB = array_pad($ldB,-5,0);
        //echo "dims=(".implode(',',$dims).")\n";
        //echo "ldA=(".implode(',',$ldA).")\n";
        //echo "ldB=(".implode(',',$ldB).")\n";
        [$dim0,$dim1,$dim2,$dim3,$dim4] = $dims;
        [$ldA0,$ldA1,$ldA2,$ldA3,$ldA4] = $ldA;
        [$ldB0,$ldB1,$ldB2,$ldB3,$ldB4] = $ldB;
        $AA = $a->buffer();
        $offsetA = $a->offset();
        $BB = $b->buffer();
        $offsetB = $b->offset();
        $CC = $c->buffer();
        $offsetC = $c->offset();
        $this->openclmath->einsum4p1(
            $dim0,
            $dim1,
            $dim2,
            $dim3,
            $dim4,
            $AA,
            $offsetA,
            $ldA0,
            $ldA1,
            $ldA2,
            $ldA3,
            $ldA4,
            $BB,
            $offsetB,
            $ldB0,
            $ldB1,
            $ldB2,
            $ldB3,
            $ldB4,
            $CC,
            $offsetC,
            $events,
            $waitEvents,
        );
        
        if($this->blocking) {
            $this->finish();
        }
        if(!isset($this->einsum4p1EquationCache[$equation])) {
            $this->einsum4p1EquationCache[$equation] = [
                $compiledLabels,
            ];
        }

        if($this->profiling) {
            $this->profilingEnd("einsum4p1");
        }
        return $c;
    }
    
    /**
     *    A(m,n,k,len) := A(m,n,k,len) : X(m,k) = True
     *                    fill         : X(m,k) = False
     */
    public function masking(
        NDArray $mask,
        NDArray $data,
        ?int $batchDims=null,
        ?int $axis=null,
        ?float $fill=null,
        ?int $mode=null,
        ?object $events=null,
        ?object $waitEvents=null 
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("masking");
        }

        if($mask->dtype()!=NDArray::bool) {
            $types = $this->dtypeToString($mask->dtype());
            throw new InvalidArgumentException('dtype of mask must be bool. :'.$types);
        }
        $batchDims ??= 0;
        if($batchDims<0) {
            $batchDims += $data->ndim();
        }
        if($batchDims>$data->ndim()) {
            throw new InvalidArgumentException(
                "batchDims ($batchDims) must be less than dims of data (".$data->ndim().") or equal."
            );
        }
        $axis ??= $batchDims;
        if($axis<0) {
            $axis += $data->ndim();
        }
        if($axis>$data->ndim()) {
            throw new InvalidArgumentException(
                "axis ($axis) must be less than to ndims of data (".$data->ndim().")"
            );
        }
        if($batchDims>$axis) {
            if($axis==0) {
                $batchDims = 0;
            } else {
                throw new InvalidArgumentException("batchDims ($batchDims) must be less than or equal to axis ($axis)");
            }
        }
        if($mask->ndim()<$batchDims) {
            throw new InvalidArgumentException(
                "batchDims ($batchDims) must be less than or equal to ndims of mask (".$mask->ndim().")"
            );
        }
        $fill ??= 0;
        $mode ??= 0;

        //echo "data  sh: ".$this->printableShapes($data->shape())."\n";
        //echo "mask sh: ".$this->printableShapes($mask->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        // data
        $outerShape = $data->shape();
        $brodacastShape = array_splice($outerShape, $batchDims);
        $innerShape = array_splice($brodacastShape, $axis-$batchDims);
        $innterBroadcastShape = array_splice($innerShape,$mask->ndim()-$batchDims);
        // mask
        $outerShapeX = $mask->shape();
        $innerShapeX = array_splice($outerShapeX, $batchDims);

        //echo "mask: ".$this->printableShapes($mask->shape())."\n";
        //echo "data: ".$this->printableShapes($data->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        //echo "axis: ".$axis."\n";
        //echo "outerShape: ".$this->printableShapes($outerShape)."\n";
        //echo "brodacastShape: ".$this->printableShapes($brodacastShape)."\n";
        //echo "innerShape: ".$this->printableShapes($innerShape)."\n";
        //echo "outerShapeX: ".$this->printableShapes($outerShapeX)."\n";
        //echo "innerShapeX: ".$this->printableShapes($innerShapeX)."\n";

        if($outerShape!=$outerShapeX) {
            throw new InvalidArgumentException('Unmatch dimension outer shape.: '.
                'mask('.implode(',',$mask->shape()).') => data('.implode(',',$data->shape()).')');
        }
        if($innerShape!=$innerShapeX) {
            throw new InvalidArgumentException('Unmatch dimension inner shape.: '.
                'mask('.implode(',',$mask->shape()).') => data('.implode(',',$data->shape()).')');
        }
        $m = (int)array_product($outerShape);
        $n = (int)array_product($brodacastShape);
        $k = (int)array_product($innerShape);
        $len = (int)array_product($innterBroadcastShape);

        //echo "\n";
        //echo "org m=$m,n=$n,k=$k,len=$len\n";
        if($n==1) {
            $k = $m*$k;
            $m = 1;
        }
        if($k==1) {
            $k = $m;
            $len = $n*$len;
            $m = 1;
            $n = 1;
        }
        //echo "adj m=$m,n=$n,k=$k,len=$len\n";

        $XX = $mask->buffer();
        $offX = $mask->offset();
        $AA = $data->buffer();
        $offA = $data->offset();

        $this->openclmath->masking(  // tempolary implements
            $m,
            $n,
            $k,
            $len,
            $fill,
            $mode,
            $XX,$offX,
            $AA,$offA,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("masking");
        }
        return $data;
    }

}