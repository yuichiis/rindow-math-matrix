<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\LinearBuffer;
use Interop\Polite\Math\Matrix\DeviceBuffer;
use Rindow\OpenCL\EventList;
use Rindow\Math\Matrix\Drivers\Service;
#use Rindow\Math\Matrix\OpenBlasBuffer;
#use Rindow\Math\Matrix\NDArrayPhp;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class LinearAlgebraCL
{
    use ComplexUtils;
    use LinalgUtils;
    use LinalgBlasCL;
    use LinalgMathCL;

    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;
    const DEFAULT_PCG32_SEQUENCE = 194625709;

    protected const WHERE_MATRIX = 0;
    protected const WHERE_VECTOR = 1;
    protected const WHERE_SCALAR = 2;

    protected Service $service;
    protected object $context;
    protected object $queue;
    protected object $blas;
    protected object $lapack;
    protected object $math;
    //protected object $blas2;
    protected object $openclmath;
    protected object $openblasmath;
    protected DeviceBuffer $rngState;
    protected int $defaultFloatType = NDArray::float32;
    /** @var array<array{array{array<int>,array<int>,array<int>}}> $einsumEquationCache */
    protected array $einsumEquationCache = [];
    /** @var array<array{array{array<int>,array<int>,array<int>}}> $einsum4p1EquationCache */
    protected array $einsum4p1EquationCache = [];
    protected bool $blocking = false;
    protected bool $scalarNumeric = false;
    protected bool $autoEvents;
    protected bool $profiling = false;
    /** @var array<string,float> $profilingStartTime */
    protected $profilingStartTime = [];
    /** @var array<string,int> $profilingCount */
    protected $profilingCount = [];
    /** @var array<string,float> $profilingTotalTime */
    protected array $profilingTotalTime = [];
    protected string $clVersion;
    protected bool $isOpenCL110;

    public function __construct(
        object $queue, Service $service, ?int $defaultFloatType=null)
    {
        $this->service = $service;
        $this->queue = $queue;
        $this->context = $queue->getContext();
        $this->blas = $service->blasCL($queue);
        $this->lapack = $service->lapack(Service::LV_ADVANCED);
        $this->math = $service->mathCLBlast($queue);
        //$this->blas2 = $service->blasCL2($queue);
        $this->openclmath = $service->mathCL($queue);
        $this->openblasmath = $service->math(Service::LV_ADVANCED);
        if($defaultFloatType!==null) {
            $this->defaultFloatType = $defaultFloatType;
        }
        $this->rngState = $this->createRngState(
            random_int(PHP_INT_MIN, PHP_INT_MAX),
            self::DEFAULT_PCG32_SEQUENCE
        );
        $this->clVersion = $this->context->getInfo(OpenCL::CL_CONTEXT_DEVICES)->getInfo(0,OpenCL::CL_DEVICE_VERSION);
        //                                                    1234567890
        $this->isOpenCL110 = substr($this->clVersion,0,10)==='OpenCL 1.1';
    }

    public function service() : Service
    {
        return $this->service;
    }

    public function getBlas() : object
    {
        return $this->blas;
    }

    public function getMath() : object
    {
        return $this->math;
    }

    public function getOpenCLMath() : object
    {
        return $this->openclmath;
    }

    public function getConfig() : string
    {
        return 'CLBlast';
    }

    public function getContext() : object
    {
        return $this->context;
    }

    public function getQueue() : object
    {
        return $this->queue;
    }

    public function getCLVersion() : string
    {
        return $this->clVersion;
    }

    public function isOpenCL110() : bool
    {
        return $this->isOpenCL110;
    }

    public function setOpenCLTestMode(int $testMode) : void
    {
        $this->openclmath->setTestMode($testMode);
    }

    public function blocking(?bool $switch=null) : bool
    {
        if($switch===null) {
            return $this->blocking;
        }
        $this->blocking = $switch;
        return $this->blocking;
    }

    public function setProfiling(bool $profiling) : void
    {
        $this->profiling = $profiling;
    }

    protected function profilingStart(string $name) : void
    {
        if(isset($this->profilingCount[$name])) {
            $this->profilingCount[$name]++;
        } else {
            $this->profilingCount[$name] = 1;
            $this->profilingTotalTime[$name] = 0;
        }
        $this->profilingStartTime[$name] = microtime(true);
    }

    protected function profilingEnd(string $name) : void
    {
        $this->profilingTotalTime[$name] +=
            microtime(true) - $this->profilingStartTime[$name];
    }

    public function profilingReport() : void
    {
        asort($this->profilingTotalTime);
        foreach($this->profilingTotalTime as $name => $time) {
            $count = $this->profilingCount[$name];
            echo sprintf("%17s:total %6e, count:%6d, average %6e\n",$name,$time,$count,$time/$count);
        }
    }

    /**
     * @param array<NDArray> $inputs
     * @return array<object|null>
     */
    public function prepareAutoEvent(
        array $inputs, ?object $events=null, ?object $waitEvents=null, ?bool $explicit=null) : array
    {
        if($this->autoEvents) {
            $autoWaitEvents = $this->newEventList();
            foreach($inputs as $array) {
                $arrayEvents = $array->getEvents();
                if($arrayEvents) {
                    $autoWaitEvents->copy($arrayEvents);
                }
            }
            if($waitEvents) {
                $autoWaitEvents->copy($waitEvents);
            }
            if($explicit) {
                $autoWaitEvents->wait();
            }
            if($events==null) {
                $events = $this->newEventList();
            }
        }
        return [$events,$waitEvents];
    }

    public function applyAutoEvent(object $output,object $events,object $waitEvents) : void
    {
        if($this->autoEvents) {
            $output->setEvents($events);
        }
    }

    public function scalarNumeric(?bool $switch=null) : bool
    {
        if($switch===null) {
            return $this->scalarNumeric;
        }
        $this->scalarNumeric = $switch;
        return $this->scalarNumeric;
    }

    public function fp64() : bool
    {
        return $this->openclmath->fp64();
    }

    public function accelerated() : bool
    {
        return true;
    }

    /**
     * @return array<string>
     */
    public function deviceTypes() : array
    {
        return $this->openclmath->deviceTypes();
    }

    public function finish() : void
    {
        $this->queue->finish();
    }

    /**
     * @return array<bool>
     */
    protected function complementTrans(?bool $trans,?bool $conj,int $dtype) : array
    {
        $trans = $trans ?? false;
        if($this->isComplex($dtype)) {
            $conj = $conj ?? $trans;
        } else {
            $conj = $conj ?? false;
        }
        return [$trans,$conj];
    }

    protected function createRngState(int $seed, int $sequence) : DeviceBuffer
    {
        $rngState = $this->alloc([2],dtype:NDArray::int64)->buffer();
        $this->openclmath->pcg32Srand($rngState, $seed, $sequence);
        return $rngState;
    }

    public function array(mixed $array, ?int $dtype=null, ?int $flags=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("array");
        }
        if($array instanceof NDArray) {
            $buffer = $array->buffer();
            if($buffer instanceof LinearBuffer) {
                ;
            } elseif($buffer instanceof DeviceBuffer) {
                return $array;
            } else {
                throw new InvalidArgumentException('Unsuppored buffer type.');
            }
        } elseif(is_array($array) || is_numeric($array) || is_bool($array) || $this->cisobject($array)) {
            $array = new NDArrayPhp($array,dtype:$dtype,service:$this->service);
        } else {
            throw new InvalidArgumentException('input value must be NDArray or array');
        }
        if($flags==null) {
            $flags = OpenCL::CL_MEM_READ_WRITE;
        }
        $flags = $flags | OpenCL::CL_MEM_COPY_HOST_PTR;
        $arrayCL = new NDArrayCL(
            $this->queue,
            $array->buffer(), dtype:$array->dtype(), shape:$array->shape(),
            offset:$array->offset(), flags:$flags, service:$this->service);
        if($this->profiling) {
            $this->profilingEnd("array");
        }
        return $arrayCL;
    }

    public function toNDArray(NDArray $ndarray) : NDArray
    {
        if($ndarray instanceof NDArrayCL) {
            return $ndarray->toNDArray();
        }
        return $ndarray;
    }

    public function scalar(mixed $array) : mixed
    {
        if($array instanceof NDArray) {
            return $array->toArray();
        }
        return $array;
    }

    /**
     * @param array<int> $shape
     */
    public function alloc(array $shape,?int $dtype=null,?int $flags=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("alloc");
        }
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        $arrayCL = new NDArrayCL(
            $this->queue, dtype:$dtype, shape:$shape,
            flags:$flags, service:$this->service);

        if($this->profiling) {
            $this->profilingEnd("alloc");
        }
        return $arrayCL;
    }

    public function zeros(
        array|NDArray $X,
        ?int $dtype=null,
        ?int $flags=null,
        ?object $events=null, ?object $waitEvents=null) : NDArray
    {
        if(is_array($X)) {
            $X = $this->alloc($X,dtype:$dtype,flags:$flags);
        }
        $value = $this->buildValByType(0.0,$X->dtype());
        $this->fill($value,$X,events:$events,waitEvents:$waitEvents);
        return $X;
    }

    public function ones(
        array|NDArray $X,
        ?int $dtype=null,
        ?int $flags=null,
        ?object $events=null, ?object $waitEvents=null) : NDArray
    {
        if(is_array($X)) {
            $X = $this->alloc($X,dtype:$dtype,flags:$flags);
        }
        if($X->dtype()==NDArray::bool) {
            $value = true;
        } else {
            $value = $this->buildValByType(1,$X->dtype());
        }
        $this->fill($value,$X,events:$events,waitEvents:$waitEvents);
        return $X;
    }

    public function zerosLike(NDArray $array,?int $flags=null) : NDArray
    {
        $newArray = $this->alloc($array->shape(),dtype:$array->dtype(),flags:$flags);
        //$events = $this->newEventList();
        $this->zeros($newArray,/*events:$events*/);
        //$events->wait();
        return $newArray;
    }

    /**
     * @param array<int> $shape
     */
    public function allocHost(array $shape, ?int $dtype=null) : NDArray
    {
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        return new NDArrayPhp(null,dtype:$dtype,shape:$shape,service:$this->service);
    }

    public function zerosHost(
        NDArray $X) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("zerosHost");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->openblasmath->zeros($N,$XX,$offX,1);
        if($this->profiling) {
            $this->profilingEnd("zerosHost");
        }
        return $X;
    }

    public function arrayHost(mixed $array, ?int $dtype=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("arrayHost");
        }
        if($array instanceof NDArray) {
            $result = $array;
        } elseif(is_array($array) || is_numeric($array) || is_bool($array) || $this->cisobject($array)) {
            $result = new NDArrayPhp($array,$dtype,service:$this->service);
        } else {
            throw new InvalidArgumentException('input value must be NDArray or array');
        }
        if($this->profiling) {
            $this->profilingEnd("arrayHost");
        }
        return $result;
    }

    #protected function newHostBuffer($size,$dtype)
    #{
    #    return new OpenBlasBuffer($size,$dtype);
    #}

    #protected function newBuffer(
    #    int $size, int $flags=null,
    #    LinearBuffer $hostBuffer=null, int $hostOffset=null,
    #    ?int $dtype=null)
    #{
    #    return $this->service->buffer()->Buffer($this->context,
    #        $size,$flags,$hostBuffer,$hostOffset,$dtype);
    #}

    public function newEventList() : object
    {
        return $this->service->openCL()->EventList();
    }

    /**
    *    ret := max |X(i)|
    */
    public function amax(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null) : float|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("amax");
        }
        $dtype = $X->dtype();
        if($this->isComplex($dtype)) {
            $rtype = ($dtype==NDArray::complex128)? NDArray::float64 : NDArray::float32;
        } else {
            $rtype = $dtype;
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$rtype,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        //$imaxEvents = $this->newEventList();
        $this->blas->iamax($N,$IRR,$offIR,$XX,$offX,1,$this->queue,/*$imaxEvents*/);
        //$imaxEvents->wait();

        $idx = $IR->toArray();
        if($this->isComplex($X->dtype())) {
            $TR = $this->alloc([],dtype:$dtype,flags:OpenCL::CL_MEM_READ_WRITE);
        } else {
            $TR = $R;
        }
        $TRR = $TR->buffer();
        $offTR = $TR->offset();
        $this->blas->copy(1,$XX,$offX+$idx,1,$TRR,$offTR,1,$this->queue,$events);
        $RR = $R->buffer();
        $offR = $R->offset();
        if($this->isComplex($dtype)) {
            $this->openclmath->absComplex(1,$TRR,$offTR,1,$RR,$offR,1,events:$events,/*$copyEvents*/);
        } else {
            $this->openclmath->abs(1,$RR,$offR,1,events:$events,/*$copyEvents*/);
        }
        //if($this->blocking) {
        //    $this->finish();
        //}
        if($this->profiling) {
            $this->profilingEnd("amax");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            return $value;
        }
        return $R;
    }

    /**
    *    ret := min |X(i)|
    */
    public function amin(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null) : float|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("amin");
        }
        $dtype = $X->dtype();
        if($this->isComplex($dtype)) {
            $rtype = ($dtype==NDArray::complex128)? NDArray::float64 : NDArray::float32;
        } else {
            $rtype = $dtype;
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$rtype,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        //$imaxEvents = $this->newEventList();
        $this->blas->iamin($N,$IRR,$offIR,$XX,$offX,1,$this->queue,/*$imaxEvents*/);
        //$imaxEvents->wait();

        $idx = $IR->toArray();
        if($this->isComplex($X->dtype())) {
            $TR = $this->alloc([],dtype:$dtype,flags:OpenCL::CL_MEM_READ_WRITE);
        } else {
            $TR = $R;
        }
        $TRR = $TR->buffer();
        $offTR = $TR->offset();
        $this->blas->copy(1,$XX,$offX+$idx,1,$TRR,$offTR,1,$this->queue,$events);
        $RR = $R->buffer();
        $offR = $R->offset();
        if($this->isComplex($dtype)) {
            $this->openclmath->absComplex(1,$TRR,$offTR,1,$RR,$offR,1,events:$events,/*$copyEvents*/);
        } else {
            $this->openclmath->abs(1,$RR,$offR,1,events:$events,/*$copyEvents*/);
        }
        //$RR = $R->buffer();
        //$offR = $R->offset();
        //$this->openclmath->reduceGather(false,false,
        //    1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingStart("amin");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            return $value;
        }
        return $R;
    }

    /**
     * g = rotg(x,y)
     */
    public function rotgxy(
        NDArray $vector,
        ?NDArray $g=null,
        ?object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rotgxy");
        }
        if($vector->shape()!=[2]) {
            throw new InvalidArgumentException("Shape of vector must be [2]: [".implode(',',$vector->shape())."]");
        }
        if($g==null) {
            $g = $this->alloc([4],dtype:$vector->dtype());
        } else {
            if($g->shape()!=[4]) {
                throw new InvalidArgumentException("Shape of g must be [4]: [".implode(',',$g->shape())."]");
            }
        }
        //$copyEvents = $this->newEventList();
        $this->copy($vector[R(0,1)],$g[R(0,1)],/*$copyEvents*/);
        $this->copy($vector[R(1,2)],$g[R(1,2)],/*$copyEvents*/);
        //$copyEvents->wait();

        if($g==null) {
            $g = $this->alloc([2],dtype:$vector->dtype());
        }
        $AA = $g->buffer();
        $offA = $g->offset();
        $BB = $g->buffer();
        $offB = $g->offset()+1;
        $CC = $g->buffer();
        $offC = $g->offset()+2;
        $SS = $g->buffer();
        $offS = $g->offset()+3;
        $this->blas->rotg(
            $AA,$offA,
            $BB,$offB,
            $CC,$offC,
            $SS,$offS,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotgxy");
        }
        return $g;
    }

    /**
    *    xy := rot(xy,g)
    */
    public function rotxy(
        NDArray $vectors,
        NDArray $g,
        ?object $events=null) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotxy");
        }
        if($vectors->ndim()!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("vectors must be 2D-NDArray: ".$shapeError." given.");
        }
        $shape = $vectors->shape()[1];
        if($shape!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("Vectors must be Vectors-NDArray: ".$shapeError." given.");
        }
        if($g->shape()!=[4]) {
            $shapeError = '['.implode(',',$g->shape()).']';
            throw new InvalidArgumentException("shape of g must be [4]: ".$shapeError." given.");
        }
        
        $N = count($vectors);
        $XX = $vectors->buffer();
        $offX = $vectors->offset();
        $YY = $vectors->buffer();
        $offY = $vectors->offset()+1;
        $CC = $g->buffer();
        $offC = $g->offset()+2;
        $SS = $g->buffer();
        $offS = $g->offset()+3;
        $this->blas->rot($N,
            $XX,$offX,2,$YY,$offY,2,
            $CC,$offC,$SS,$offS
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotxy");
        }
    }

    /**
     * d1,d2,b1,p = rotmg(x,y)   b1: rotated x   p: params  d1,d2:works
     */
    public function rotmgxy(
        NDArray $vector,
        ?NDArray $d=null,
        ?NDArray $g=null,
        ?object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rotmgxy");
        }
        if($vector->shape()!=[2]) {
            throw new InvalidArgumentException("Shape of vector must be [2]: [".implode(',',$vector->shape())."]");
        }
        if($d==null) {
            $d = $this->ones($this->alloc([2],dtype:$vector->dtype()));
        } else {
            if($d->shape()!=[2]) {
                throw new InvalidArgumentException("Shape of d must be [2]: [".implode(',',$d->shape())."]");
            }
        }
        if($g==null) {
            $g = $this->zeros($this->alloc([6],dtype:$vector->dtype()));
        } else {
            if($g->shape()!=[6]) {
                throw new InvalidArgumentException("Shape of g must be [6]: [".implode(',',$g->shape())."]");
            }
        }
        $B2 = $this->alloc([1],dtype:$vector->dtype());
        //$copyEvents = $this->newEventList();
        $this->copy($vector[R(0,1)],$g[R(0,1)],/*$copyEvents*/);
        $this->copy($vector[R(1,2)],$B2,/*$copyEvents*/);
        //$copyEvents->wait();

        $DD1 = $d->buffer();
        $offD1 = $d->offset();
        $DD2 = $d->buffer();
        $offD2 = $d->offset()+1;
        $BB1 = $g->buffer();
        $offB1 = $g->offset();
        $BB2 = $B2->buffer();
        $offB2 = $B2->offset();
        $PP = $g->buffer();
        $offP = $g->offset()+1;
        $this->blas->rotmg(
            $DD1,$offD1,
            $DD2,$offD2,
            $BB1,$offB1,
            $BB2,$offB2,
            $PP,$offP,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotmgxy");
        }
        return $g;
    }

    /**
    *    x,y := rot(x,y,p)
    */
    public function rotmxy(
        NDArray $vectors,
        NDArray $g,
        ?object $events=null) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotmxy");
        }
        if($vectors->ndim()!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("vectors must be 2D-NDArray: ".$shapeError." given.");
        }
        $shape = $vectors->shape()[1];
        if($shape!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("Vectors must be Vectors-NDArray: ".$shapeError." given.");
        }
        if($g->shape()!=[6]) {
            $shapeError = '['.implode(',',$g->shape()).']';
            throw new InvalidArgumentException("shape of g must be [6]: ".$shapeError." given.");
        }

        $N = count($vectors);
        $XX = $vectors->buffer();
        $offX = $vectors->offset();
        $YY = $vectors->buffer();
        $offY = $vectors->offset()+1;
        $PP = $g->buffer();
        $offP = $g->offset()+1;
        $this->blas->rotm(
            $N,
            $XX,$offX,2,
            $YY,$offY,2,
            $PP,$offP,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotmxy");
        }
    }

    /**
    *
    */
    public function matmul(
        NDArray $A,
        NDArray $B,
        ?bool $transA=null,
        ?bool $transB=null,
        ?NDArray $C=null,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?bool $conjA=null,
        ?bool $conjB=null,
        ?object $events=null
        ) : NDArray
    {
        [$transA,$conjA] = $this->complementTrans($transA,$conjA,$A->dtype());
        [$transB,$conjB] = $this->complementTrans($transB,$conjB,$B->dtype());
        if($this->profiling) {
            $this->profilingStart("matmul");
        }
        if($A->ndim()<2 || $B->ndim()<2) {
            throw new InvalidArgumentException('Dimensions rank must be greater then 2D or equal:['.
                implode(',',$A->shape()).']<=>['.implode(',',$B->shape()).']');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        $shapeEA = [array_pop($shapeA)];
        array_unshift($shapeEA,array_pop($shapeA));
        $shapeEB = [array_pop($shapeB)];
        array_unshift($shapeEB,array_pop($shapeB));
        $batchA = (int)array_product($shapeA);
        $batchB = (int)array_product($shapeB);
        $flatA = $A->reshape(array_merge([$batchA],$shapeEA));
        $flatB = $B->reshape(array_merge([$batchB],$shapeEB));

        if($transA) {
            $shapeEA = array_reverse($shapeEA);
        }
        if($transB) {
            $shapeEB = array_reverse($shapeEB);
        }
        if($shapeEA[1]!=$shapeEB[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and the number of rows in "B" must be the same:['.
                implode(',',$A->shape()).']<=>['.implode(',',$B->shape()).']');
        }

        $AA = $A->buffer();
        $BB = $B->buffer();
        $M = $shapeEA[0];
        $N = $shapeEB[1];
        $K = $shapeEA[1];

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = $this->transToCode($transA,$conjA);
        $transB = $this->transToCode($transB,$conjB);

        $shapeEC = [$shapeEA[0],$shapeEB[1]];
        if($batchA>$batchB) {
            $broadcastDest = $batchA;
            $broadcastBase = $batchB;
            $orgShapeC=array_merge($shapeA,$shapeEC);
        } else {
            $broadcastDest = $batchB;
            $broadcastBase = $batchA;
            $orgShapeC=array_merge($shapeB,$shapeEC);
        }
        if($broadcastDest % $broadcastBase != 0) {
            throw new InvalidArgumentException('Matrix size-incompatible for broadcast:['.
                implode(',',$A->shape()).']<=>['.implode(',',$B->shape()).']');
        }
        if($C!=null) {
            if($C->shape()!=$orgShapeC) {
                throw new InvalidArgumentException('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns:['.
                    implode(',',$A->shape()).'] , ['.implode(',',$B->shape()).'] => ['.implode(',',$C->shape()).']');
            }
        } else {
            $C = $this->alloc($orgShapeC,dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            //$zerosEvents = $this->newEventList();
            $this->zeros($C,/*events:$zerosEvents*/);
            //$zerosEvents->wait();
        }
        $flatC = $C->reshape(array_merge([$broadcastDest],$shapeEC));
        $CC = $C->buffer();
        $repeats = intdiv($broadcastDest,$broadcastBase);
        $offA = $A->offset();
        $offB = $B->offset();
        $offC = $C->offset();
        $incA = $M*$K;
        $incB = $N*$K;
        $incC = $M*$N;
        for($i=0;$i<$repeats;$i++) {
            if($batchA>$batchB) {
                $offB = $B->offset();
            } else {
                $offA = $A->offset();
            }
            $this->math->gemmStridedBatched(
                BLAS::RowMajor,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,$incA,
                $BB,$offB,$ldb,$incB,
                $beta,
                $CC,$offC,$ldc,$incC,
                $broadcastBase,
                $this->queue,$events
            );
            $offA += $incA*$broadcastBase;
            $offB += $incB*$broadcastBase;
            $offC += $incC*$broadcastBase;
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("matmul");
        }
        return $C;
    }

    /**
    *    ret := max X(i)
    */
    public function max(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("max");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        if($this->isFloat($X)) {
            $this->math->imax($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);
        } else {
            $this->openclmath->reduceArgMax(
                1,          // $m
                $N,         // $n
                1,          // $k
                $XX, $offX,
                $IRR, $offIR,
                $imaxEvents
            );
        }

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->reduceGather(false,false,
            1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("max");
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := min X(i)
    */
    public function min(
        NDArray $X,
        ?NDArray $R=null,
        ?object $events=null) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("min");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        if($this->isFloat($X)) {
            $this->math->imin($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);
        } else {
            $this->openclmath->imin($N,$IRR,$offIR,$XX,$offX,1,$imaxEvents);
            //throw new InvalidArgumentException("Unsuppored data type. Only float types are supported.");
        }

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->reduceGather(false,false,
            1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("min");
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
     * @param array<NDArray> $values
     */
    public function stack(
        array $values,
        ?int $axis=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("stack");
        }
        if($axis==null){
            $axis=0;
        }
        if($axis==0){
            $m = count($values);
            $shape = $values[0]->shape();
            array_unshift($shape,$m);
            $output = $this->alloc($shape,dtype:$values[0]->dtype());
            $i = 0;
            foreach($values as $value){
                if(!($value instanceof NDArray)) {
                    throw new InvalidArgumentException('values must be array of NDArray');
                }
                $shape = $value->shape();
                array_unshift($shape,1);
                $value = $value->reshape(
                    $shape);
                $this->doSlice(true,
                    $output,
                    [$i],[1],
                    $value,
                    $events,$waitEvents
                );
                $i++;
            }
        } elseif($axis==1){
            $n = count($values);
            $shape = $values[0]->shape();
            $m = array_shift($shape);
            array_unshift($shape,$n);
            array_unshift($shape,$m);
            $output = $this->alloc($shape,dtype:$values[0]->dtype());
            $i = 0;
            foreach($values as $value){
                if(!($value instanceof NDArray)) {
                    throw new InvalidArgumentException('values must be array of NDArray');
                }
                $shape = $value->shape();
                $m = array_shift($shape);
                array_unshift($shape,1);
                array_unshift($shape,$m);
                $value = $value->reshape(
                    $shape);
                $this->doSlice(true,
                    $output,
                    [0,$i],[-1,1],
                    $value,
                    $events,$waitEvents
                );
                $i++;
            }
        } elseif($axis==2){
            $k = count($values);
            $shape = $values[0]->shape();
            $m = array_shift($shape);
            $n = array_shift($shape);
            array_unshift($shape,$k);
            array_unshift($shape,$n);
            array_unshift($shape,$m);
            $output = $this->alloc($shape,dtype:$values[0]->dtype());
            $i = 0;
            foreach($values as $value){
                if(!($value instanceof NDArray)) {
                    throw new InvalidArgumentException('values must be array of NDArray');
                }
                $shape = $value->shape();
                $m = array_shift($shape);
                $n = array_shift($shape);
                array_unshift($shape,1);
                array_unshift($shape,$n);
                array_unshift($shape,$m);
                $value = $value->reshape(
                    $shape);
                $this->doSlice(true,
                    $output,
                    [0,0,$i],[-1,-1,1],
                    $value,
                    $events,$waitEvents
                );
                $i++;
            }
        } else {
            throw new InvalidArgumentException('unsuppoted axis');
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("stack");
        }
        return $output;
    }

    /**
     * @param array<NDArray> $values
     */
    public function concat(
        array $values,
        ?int $axis=null,
        ?object $events=null, ?object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("concat");
        }
        if($axis===null) {
            $axis = -1;
        }
        if($axis<0) {
            $axis = $values[0]->ndim() + $axis;
        }
        $m = null;
        $base = null;
        $n = 0;
        $reshapeValues = [];
        $shape = [];
        $shapePrefix = [];
        foreach ($values as $value) {
            $shape = $value->shape();
            //$shapePrefix = [];
            //for($j=0;$j<$axis;$j++) {
            //    $shapePrefix[] = array_shift($shape);
            //}
            $shapePrefix = array_splice($shape,0,$axis);
            $mm = (int)array_product($shapePrefix);
            $nn = array_shift($shape);
            if($base===null) {
                $m = $mm;
                $base = $shape;
            } else {
                if($m!=$mm||$base!=$shape) {
                    throw new InvalidArgumentException('Unmatch shape: '.
                        $this->printableShapes($values));
                    }
            }
            $n += $nn;
            $reshapeValues[] = $value->reshape(array_merge([$mm,$nn],$shape));
        }
        $dims = $shape;
        $shape = array_merge([$m,$n],$shape);
        $output = $this->alloc($shape,dtype:$values[0]->dtype());
        $i = 0;
        foreach ($reshapeValues as $value) {
            $nn = $value->shape()[1];
            $this->doSlice(true,
                $output,
                [0,$i],[-1,$nn],
                $value,
                $events,$waitEvents
            );
            $i += $nn;
        }
        $output = $output->reshape(array_merge($shapePrefix,[$n],$dims));
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("concat");
        }
        return $output;
    }

    /**
     * @param array<int> $sizeSplits
     * @return array<NDArray>
     */
    public function split(
        NDArray $input, array $sizeSplits, ?int $axis=null,
        ?object $events=null, ?object $waitEvents=null
        ) : array
    {
        if($this->profiling) {
            $this->profilingStart("split");
        }
        if($axis===null) {
            $axis = -1;
        }
        if($axis<0) {
            $axis = $input->ndim() + $axis;
        }
        $shape = $input->shape();
        //$shapePrefix = [];
        //for($j=0;$j<$axis;$j++) {
        //    $shapePrefix[] = array_shift($shape);
        //}
        $shapePrefix = array_splice($shape,0,$axis);
        $m = (int)array_product($shapePrefix);
        $n = array_shift($shape);
        $input = $input->reshape(array_merge([$m,$n],$shape));
        $outputs = [];
        $dtype = $input->dtype();
        foreach($sizeSplits as $size) {
            $outputs[] = $this->alloc(array_merge($shapePrefix,[$size],$shape),dtype:$dtype);
        }
        $i = 0;
        $outidx = 0;
        foreach($sizeSplits as $size) {
            $this->doSlice(false,
                $input,
                [0,$i],[-1,$size],
                $outputs[$outidx]->reshape(array_merge(
                    [(int)array_product($shapePrefix)],
                    array_merge([$size],$shape)
                )),
                $events,$waitEvents
            );//->reshape(array_merge($shapePrefix,[$size],$shape));
            $i += $size;
            $outidx++;
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("split");
        }
        return $outputs;
    }

    /**
     * @return array<NDArray>
     */
    public function svd(NDArray $matrix, ?bool $fullMatrices=null) : array
    {
        if($this->profiling) {
            $this->profilingStart("svd");
        }
        if($matrix->ndim()!=2) {
            throw new InvalidArgumentException("input array must be 2D array");
        }
        if($fullMatrices===null)
            $fullMatrices = true;
        [$m,$n] = $matrix->shape();
        if($fullMatrices) {
            $jobu  = 'A';
            $jobvt = 'A';
            $ldA = $n;
            $ldU = $m;
            $ldVT = $n;
        } else {
            $jobu  = 'S';
            $jobvt = 'S';
            $ldA = $n;
            $ldU = min($m,$n);
            #$ldVT = min($m,$n);
            $ldVT = $n; // bug in the lapacke ???
        }

        $S = $this->allocHost([min($m,$n)],$matrix->dtype());
        $this->zerosHost($S);
        $U = $this->allocHost([$m,$ldU],$matrix->dtype());
        $this->zerosHost($U);
        $VT = $this->allocHost([$ldVT,$n],$matrix->dtype());
        $this->zerosHost($VT);
        $SuperB = $this->allocHost([min($m,$n)-1],$matrix->dtype());
        $this->zerosHost($SuperB);

        $hostMatrix = $matrix->toNDArray();

        $AA = $hostMatrix->buffer();
        $offsetA = $hostMatrix->offset();
        $SS = $S->buffer();
        $offsetS = $S->offset();
        $UU = $U->buffer();
        $offsetU = $U->offset();
        $VVT = $VT->buffer();
        $offsetVT = $VT->offset();
        $SuperBB = $SuperB->buffer();
        $offsetSuperB = $SuperB->offset();
        $this->lapack->gesvd(
            self::LAPACK_ROW_MAJOR,
            ord($jobu),
            ord($jobvt),
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );
        $U = $this->array($U);
        $S = $this->array($S);
        $VT = $this->array($VT);
        if(!$fullMatrices) {
            // bug in the lapacke ???
            //$copyEvents = $this->newEventList();
            $VT = $this->copy($VT[R(0,min($m,$n))],null,/*$copyEvents*/);
            //$copyEvents->wait();
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("svd");
        }
        return [$U,$S,$VT];
    }

    public function solveg(NDArray $a, NDArray $b, ?float $epsilon=null) : NDArray
    {
        if($epsilon===null) {
            $epsilon = 1e-7;
        }
        if($a->ndim()!=2) {
            throw new InvalidArgumentException('matrix a must be 2d');
        }
        if($b->ndim()!=1) {
            throw new InvalidArgumentException('matrix a must be 1d');
        }
        $shapeA = $a->shape();
        $shapeB = $b->shape();
        if($shapeA[0]!=$shapeB[0]) {
            throw new InvalidArgumentException('Unmatch shape A rows and B: '.$shapeA[0].'!='.$shapeB[0]);
        }
        if($a->dtype()!=$b->dtype()) {
            throw new InvalidArgumentException('A and B must be same data type');
        }
        $m = $shapeA[0];
        $n = $shapeA[1];
        for($j = 0; $j < $n; ++$j) {
            for($i = $j+1; $i < $m; ++$i) {
                $pa = $a[$j][[$j,$j+1]];
                $pb = $a[$i][[$j,$j+1]];
                if(abs($pb->toArray()[0]) < $epsilon) {
                    continue;
                }
                [$r,$z,$c,$s] = $this->rotg($pa,$pb);

                $num_elements_A = $n - $j;
                $rowA_j_start = $a[$j][[$j,$j+$num_elements_A]]; // j行目のj列目から
                $rowA_i_start = $a[$i][[$j,$j+$num_elements_A]]; // i行目のj列目から
                $this->rot($rowA_j_start, $rowA_i_start, $c, $s);
                $this->rot($b[[$j,$j+1]], $b[[$i,$i+1]], $c, $s);
            }
        }

        $x = $this->alloc([$n],$a->dtype());

        for ($i=$n-1; $i>=0; --$i) {
            if (abs($a[$i][[$i,$i+1]]->toArray()[0]) < $epsilon) {
                throw new InvalidArgumentException("Matrix is singular or near-singular (zero pivot encountered at R[$i][$i]).");
            }
            $sum = 0;
            for ($k = $i+1; $k<$n; ++$k) {
                $sum += $a[$i][$k] * $x[$k];
            }
    
            // x[i] = (b'[i] - sum) / R[i][i]
            $x[$i] = ($b[$i] - $sum) / $a[$i][$i];
        }

        return $x;
    }
}
