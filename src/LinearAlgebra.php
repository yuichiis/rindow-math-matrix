<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;
use InvalidArgumentException;
use RuntimeException;
use LogicException;
use Rindow\Math\Matrix\Drivers\Service;
use function Rindow\Math\Matrix\R;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpPcg32;

class LinearAlgebra
{
    use ComplexUtils;
    use LinalgUtils;
    use LinalgBlas;
    use LinalgMath;

    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;
    const DEFAULT_PCG32_SEQUENCE = 194625709;

    protected const WHERE_MATRIX = 0;
    protected const WHERE_VECTOR = 1;
    protected const WHERE_SCALAR = 2;

    protected bool $iaminwarning = false;
    protected Service $service;
    protected object $blas;
    protected object $lapack;
    protected object $math;
    protected Buffer|PhpPcg32 $rngState;
    protected int $defaultFloatType = NDArray::float32;
    /** @var array<array{array{array<int>,array<int>,array<int>}}> $einsumEquationCache */
    protected array $einsumEquationCache = [];
    /** @var array<array{array{array<int>,array<int>,array<int>}}> $einsum4p1EquationCache */
    protected array $einsum4p1EquationCache = [];
    protected bool $profiling = false;
    /** @var array<string,float> $profilingStartTime */
    protected $profilingStartTime = [];
    /** @var array<string,int> $profilingCount */
    protected $profilingCount = [];
    /** @var array<string,float> $profilingTotalTime */
    protected array $profilingTotalTime = [];
    
    public function __construct(
        Service $service,
        ?int $defaultFloatType=null,
        ?int $serviceLevel=null)
    {
        $this->service = $service;
        $this->blas = $service->blas($serviceLevel);
        $this->lapack = $service->lapack($serviceLevel);
        $this->math = $service->math($serviceLevel);
        if($defaultFloatType!==null) {
            $this->defaultFloatType = $defaultFloatType;
        }
        $this->rngState = $this->createRngState(
            random_int(PHP_INT_MIN, PHP_INT_MAX),
            self::DEFAULT_PCG32_SEQUENCE
        );
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

    public function getConfig() : string
    {
        return $this->blas->getConfig();
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

    public function fp64() : bool
    {
        return true;
    }

    public function accelerated() : bool
    {
        return false;
    }

    public function finish() : void
    {
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

    protected function createRngState(int $seed, int $sequence) : Buffer|PhpPcg32
    {
        if($this->math instanceof \Rindow\Math\Matrix\Drivers\MatlibPHP\PhpMath) {
            $rngState = $this->math->createPcg32State($seed,$sequence);
        } else {
            $rngState = $this->alloc([2],dtype:NDArray::int64)->buffer();
            $this->math->pcg32Srand($rngState, $seed, $sequence);
        }
        return $rngState;
    }

    public function array(mixed $array, ?int $dtype=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("array");
        }
        if($array instanceof NDArray) {
            $result = $array;
        } elseif(is_array($array) || is_numeric($array) || is_bool($array) || $this->cisobject($array)) {
            $result = new NDArrayPhp($array,$dtype,service:$this->service);
        } else {
            throw new InvalidArgumentException('input value must be NDArray or array');
        }
        if($this->profiling) {
            $this->profilingEnd("array");
        }
        return $result;
    }

    public function toNDArray(NDArray $ndarray) : NDArray
    {
        if(!($ndarray instanceof NDArrayPhp)) {
            throw new InvalidArgumentException('input value must be NDArrayPhp');
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
    public function alloc(array $shape,?int $dtype=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("alloc");
        }
        if($dtype===null) {
            $dtype = $this->defaultFloatType;
        }
        $result = new NDArrayPhp(null,$dtype,$shape,service:$this->service);
        if($this->profiling) {
            $this->profilingEnd("alloc");
        }
        return $result;
    }

    public function ones(
        array|NDArray $X,
        ?int $dtype=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("ones");
        }
        if(is_array($X)) {
            $X = $this->alloc($X,dtype:$dtype);
        }
        if($X->dtype()==NDArray::bool) {
            $value = true;
        } else {
            $value = $this->buildValByType(1,$X->dtype());
        }
        $this->fill($value,$X);
        return $X;
    }

    public function zerosLike(NDArray $array) : NDArray
    {
        $newArray = $this->alloc($array->shape(),dtype:$array->dtype());
        $this->zeros($newArray);
        return $newArray;
    }

    /**
    *    ret := max |X(i)|
    */
    public function amax(
        NDArray $X) : float
    {
        if($this->profiling) {
            $this->profilingStart("amax");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->blas->iamax($N,$XX,$offX,1);
        if($this->isComplex($X->dtype())) {
            $result = $this->cabs($XX[$offX+$i]);
        } else {
            $result = abs($XX[$offX+$i]);
        }
        if($this->profiling) {
            $this->profilingEnd("amax");
        }
        return $result;
    }

    /**
    *    ret := min |X(i)|
    */
    public function amin(
        NDArray $X) : float
    {
        if($this->profiling) {
            $this->profilingStart("amin");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($this->blas->hasIamin()) {
            $i = $this->blas->iamin($N,$XX,$offX,1);
        } else {
            $i = $this->blasIaminCompatible($N,$XX,$offX,1);
        }
        if($this->isComplex($X->dtype())) {
            $result = $this->cabs($XX[$offX+$i]);
        } else {
            $result = abs($XX[$offX+$i]);
        }
        if($this->profiling) {
            $this->profilingEnd("amin");
        }
        return $result;
    }

    /**
     * g = rotg(x,y)
     */
    public function rotgxy(
        NDArray $vector,
        ?NDArray $g=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rotgxy");
        }
        if($vector->shape()!=[2]) {
            throw new InvalidArgumentException("Shape of vector must be (2): ".$this->shapeToString($vector->shape())." given.");
        }
        if($g==null) {
            $g = $this->alloc([4],dtype:$vector->dtype());
        } else {
            if($g->shape()!=[4]) {
                throw new InvalidArgumentException("Shape of g must be (4): ".$this->shapeToString($g->shape())." given.");
            }
        }
        $this->copy($vector[R(0,1)],$g[R(0,1)]);
        $this->copy($vector[R(1,2)],$g[R(1,2)]);
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
            $SS,$offS
        );
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
        ) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotxy");
        }
        if($vectors->ndim()!=2) {
            $shapeError = $this->shapeToString($vectors->shape());
            throw new InvalidArgumentException("vectors must be 2D-NDArray: ".$shapeError." given.");
        }
        $shape = $vectors->shape()[1];
        if($shape!=2) {
            $shapeError = $this->shapeToString($vectors->shape());
            throw new InvalidArgumentException("Vectors must be Vectors-NDArray: ".$shapeError." given.");
        }
        if($g->shape()!=[4]) {
            $shapeError = $this->shapeToString($g->shape());
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
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rotmgxy");
        }
        if($vector->shape()!=[2]) {
            throw new InvalidArgumentException("Shape of vector must be (2): ".$this->shapeToString($vector->shape())." given.");
        }
        if($d==null) {
            $d = $this->ones($this->alloc([2],dtype:$vector->dtype()));
        } else {
            if($d->shape()!=[2]) {
                throw new InvalidArgumentException("Shape of d must be (2): ".$this->shapeToString($d->shape())." given.");
            }
        }
        if($g==null) {
            $g = $this->zeros($this->alloc([6],dtype:$vector->dtype()));
        } else {
            if($g->shape()!=[6]) {
                throw new InvalidArgumentException("Shape of g must be [6]: ".$this->shapeToString($g->shape())." given.");
            }
        }
        $this->copy($vector[R(0,1)],$g[R(0,1)]);

        $DD1 = $d->buffer();
        $offD1 = $d->offset();
        $DD2 = $d->buffer();
        $offD2 = $d->offset()+1;
        $BB1 = $g->buffer();
        $offB1 = $g->offset();
        $BB2 = $vector->buffer();
        $offB2 = $vector->offset()+1;
        $PP = $g->buffer();
        $offP = $g->offset()+1;
        $this->blas->rotmg(
            $DD1,$offD1,
            $DD2,$offD2,
            $BB1,$offB1,
            $BB2,$offB2,
            $PP,$offP
        );
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
        ) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotmxy");
        }
        if($vectors->ndim()!=2) {
            $shapeError = $this->shapeToString($vectors->shape());
            throw new InvalidArgumentException("vectors must be 2D-NDArray: ".$shapeError." given.");
        }
        $shape = $vectors->shape()[1];
        if($shape!=2) {
            $shapeError = $this->shapeToString($vectors->shape());
            throw new InvalidArgumentException("Vectors must be Vectors-NDArray: ".$shapeError." given.");
        }
        if($g->shape()!=[6]) {
            $shapeError = $this->shapeToString($g->shape());
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
        );
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
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("matmul");
        }
        [$transA,$conjA] = $this->complementTrans($transA,$conjA,$A->dtype());
        [$transB,$conjB] = $this->complementTrans($transB,$conjB,$B->dtype());
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
            $C = $this->zeros($this->alloc($orgShapeC,dtype:$A->dtype()));
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
            for($j=0;$j<$broadcastBase;$j++) {
                $this->blas->gemm(
                    BLAS::RowMajor,$transA,$transB,
                    $M,$N,$K,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb,
                    $beta,
                    $CC,$offC,$ldc);
                $offA+=$incA;
                $offB+=$incB;
                $offC+=$incC;
            }
        }
        if($this->profiling) {
            $this->profilingEnd("matmul");
        }
        return $C;
    }

    //protected function createIdentityMatrix(int $n, int $dtype) : Buffer
    //{
    //    if ($n <= 0) {
    //        throw new InvalidArgumentException("Error: Matrix dimension must be positive.");
    //    }
    //    // Use calloc for zero initialization, which works for complex zero (0.0 + 0.0i)
    //    // assuming the underlying representation of 0.0f is all bits zero.
    //    $imat = $this->zeros($this->alloc([$n,$n],dtype:$dtype))->buffer();
//
    //    // Set diagonal elements to 1.0 + 0.0i
    //    // Accessing column-major matrix I(n x n): Element (i, i) is at index i + i * n
    //    for($i=0; $i<$n; $i++) {
    //        $imat[$i + $i * $n] = $this->buildValByType(1.0,$dtype); // I is the imaginary unit from complex.h
    //    }
    //    return $imat;
    //}
    //
    //protected function omatcopyCompatible(
    //    int $order,
    //    int $trans,
    //    int $rows,
    //    int $cols,
    //    float|object $alpha,
    //    Buffer $A, int $offsetA, int $ldA,
    //    Buffer $B, int $offsetB, int $ldB,
    //) : void
    //{
    //    if ($rows <= 0 || $cols <= 0) {
    //        return; // Nothing to do for empty matrices
    //    }
    //    $dtype = $A->dtype();
    //
    //    // Beta parameter for GEMM: C = alpha*op(A)*Id + beta*C
    //    // We want B = alpha*op(A), so we use beta = 0.
    //    $beta = $this->buildValByType(0.0,$dtype);
//
    //    // int m, n, k;           // Dimensions for GEMM C(m x n) = op(A)(m x k) * Identity(k x n)
    //    // int ld_identity;       // Leading dimension of the identity matrix
    //
    //    // Determine GEMM parameters based on the transpose operation
    //    if ($trans == BLAS::NoTrans) {
    //        // B(rows x cols) = alpha * A(rows x cols) * I(cols x cols)
    //        $m = $rows;          // Rows of B and A
    //        $n = $cols;          // Columns of B and A
    //        $k = $cols;          // Inner dimension (columns of A, rows of I)
    //        $ldIdentity = $k;    // Identity is k x k = cols x cols
    //
    //        $identity = $this->createIdentityMatrix($k,$dtype);
    //
    //        // C = alpha * op(A) * op(B) + beta * C
    //        // Here: B_out = alpha * A * I + 0 * B_out
    //        $this->blas->gemm($order, BLAS::NoTrans, BLAS::NoTrans, // op(A)=A, op(Id)=Id
    //                    $m, $n, $k,
    //                    $alpha, $A, $offsetA, $ldA,
    //                    $identity, 0, $ldIdentity, // Matrix B in GEMM is Identity
    //                    $beta, $B, $offsetB, $ldB); // Matrix C in GEMM is output B
    //
    //    } else if ($trans == BLAS::Trans) {
    //        // B(cols x rows) = alpha * A^T(cols x rows) * I(rows x rows)
    //        $m = $cols;          // Rows of B and A^T
    //        $n = $rows;          // Columns of B and A^T
    //        $k = $rows;          // Inner dimension (columns of A^T, rows of I)
    //        $ldIdentity = $k;    // Identity is k x k = rows x rows
    //        $identity = $this->createIdentityMatrix($k,$dtype);
    //
    //        // Here: B_out = alpha * A^T * I + 0 * B_out
    //        $this->blas->gemm($order, BLAS::Trans, BLAS::NoTrans, // op(A)=A^T, op(Id)=Id
    //                    $m, $n, $k,
    //                    $alpha, $A, $offsetA, $ldA,  // A is input, operation is Transpose
    //                    $identity, 0, $ldIdentity,
    //                    $beta, $B, $offsetB, $ldB);
    //
    //    } else if ($trans == BLAS::ConjTrans) {
    //        // B(cols x rows) = alpha * A^H(cols x rows) * I(rows x rows)
    //        $m = $cols;          // Rows of B and A^H
    //        $n = $rows;          // Columns of B and A^H
    //        $k = $rows;          // Inner dimension (columns of A^H, rows of I)
    //        $ldIdentity = $k;    // Identity is k x k = rows x rows
    //
    //        $identity = $this->createIdentityMatrix($k,$dtype);
    //
    //        // Here: B_out = alpha * A^H * I + 0 * B_out
    //        $this->blas->gemm($order, BLAS::ConjTrans, BLAS::NoTrans, // op(A)=A^H, op(Id)=Id
    //                    $m, $n, $k,
    //                    $alpha, $A, $offsetA, $ldA, // A is input, operation is Conjugate Transpose
    //                    $identity, 0, $ldIdentity,
    //                    $beta, $B, $offsetB, $ldB);
    //    } else {
    //        throw new InvalidArgumentException("Error: Invalid transpose parameter provided.");
    //        // No operation performed
    //    }
    //}


    /**
    *    ret := max X(i)
    */
    public function max(
        NDArray $X) : float
    {
        if($this->profiling) {
            $this->profilingStart("max");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->math->imax($N,$XX,$offX,1);
        $result = $XX[$offX+$i];
        if($this->profiling) {
            $this->profilingEnd("max");
        }
        return $result;
    }

    /**
    *    ret := min X(i)
    */
    public function min(
        NDArray $X) : float
    {
        if($this->profiling) {
            $this->profilingStart("min");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->math->imin($N,$XX,$offX,1);
        $result = $XX[$offX+$i];
        if($this->profiling) {
            $this->profilingEnd("min");
        }
        return $result;
    }

    /**
     * @param array<NDArray> $values
     */
    public function stack(
        array $values,
        ?int $axis=null
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
                    $value
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
                    $value
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
                    $value
                );
                $i++;
            }
        } else {
            throw new InvalidArgumentException('unsuppoted axis');
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
        ?int $axis=null
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
                $value
            );
            $i += $nn;
        }
        $output = $output->reshape(array_merge($shapePrefix,[$n],$dims));
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
        NDArray $input, array $sizeSplits, ?int $axis=null
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
        $i = 0;
        $outputs = [];
        foreach ($sizeSplits as $size) {
            $outputs[] = $this->doSlice(false,
                $input,
                [0,$i],[-1,$size]
            )->reshape(array_merge($shapePrefix,[$size],$shape));
            $i += $size;
        }
        if($this->profiling) {
            $this->profilingEnd("split");
        }
        return $outputs;
    }

    /**
     * @return array<NDArray>
     */
    public function svd(
        NDArray $matrix,
        ?bool $fullMatrices=null) : array
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
        $k = min($m, $n);
        if($fullMatrices) {
            $jobu  = 'A';
            $jobvt = 'A';
            $rowsU = $m; $colsU = $m;    // U is m x m
            $rowsVT = $n; $colsVT = $n; // VT is n x n
        } else {
            $jobu  = 'S';
            $jobvt = 'S';
            $rowsU = $m; $colsU = $k;    // U is m x k
            //$rowsVT = $k; $colsVT = $n;// VT is k x n
            $rowsVT = $n; $colsVT = $n;  // bug int the lapacke
        }
        $ldA = $n;       // Input A is ROW_MAJOR (m x n), ld = n
        $ldU = $colsU;   // Output U is ROW_MAJOR (rowsU x colsU), ld = colsU
        $ldVT = $colsVT; // Output VT is ROW_MAJOR (rowsVT x colsVT), ld = colsVT = n

        $S = $this->alloc([$k],dtype:$matrix->dtype());
        $this->zeros($S);
        // Allocate U and VT with correct ROW_MAJOR dimensions and leading dimensions
        $U = $this->alloc([$rowsU, $colsU],dtype:$matrix->dtype());
        $this->zeros($U);
        $VT = $this->alloc([$rowsVT, $colsVT],dtype:$matrix->dtype());
        $this->zeros($VT);
        $SuperB = $this->alloc([$k-1],dtype:$matrix->dtype()); // Size is min(m,n)-1
        $this->zeros($SuperB);

        $AA = $matrix->buffer();
        $offsetA = $matrix->offset();
        $SS = $S->buffer();
        $offsetS = $S->offset();
        $UU = $U->buffer();
        $offsetU = $U->offset();
        $VVT = $VT->buffer();
        $offsetVT = $VT->offset();
        $SuperBB = $SuperB->buffer();
        $offsetSuperB = $SuperB->offset();
        $this->lapack->gesvd(
            self::LAPACK_ROW_MAJOR, // Specify layout
            ord($jobu),
            ord($jobvt),
            $m,
            $n,
            $AA,  $offsetA,  $ldA,  // Pass ROW_MAJOR A and its ld
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,  // Pass ROW_MAJOR U buffer and its ld
            $VVT, $offsetVT, $ldVT, // Pass ROW_MAJOR VT buffer and its ld
            $SuperBB,  $offsetSuperB
        );
        if(!$fullMatrices) {
            // bug in the lapacke ???
            $VT = $this->copy($VT[R(0,$k)]);
        }
        if($this->profiling) {
            $this->profilingEnd("svd");
        }
        return [$U,$S,$VT];
    }

    /**
     *     X := isfinite(X)
     */
    public function isfinite(
        NDArray $X
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("isfinite");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($this->math instanceof \Rindow\Math\Matrix\Drivers\MatlibPHP\PhpMath) {
            $this->math->isfinite(
                $n,
                $XX,$offX,1
            );
        } else {
            $greater = $this->greater($this->copy($X),-INF);
            $this->less($X,INF);
            $this->multiply($greater,$X);
        }
        if($this->profiling) {
            $this->profilingEnd("isfinite");
        }
        return $X;
    }

    /**
     *     X := isfinite(X)
     */
    public function isinf(
        NDArray $X
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("isinf");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($this->math instanceof \Rindow\Math\Matrix\Drivers\MatlibPHP\PhpMath) {
            $this->math->isinf(
                $n,
                $XX,$offX,1
            );
        } else {
            $greater = $this->greaterEqual($this->copy($X),INF);
            $this->lessEqual($X,-INF);
            $this->axpy($greater,$X);
        }

        if($this->profiling) {
            $this->profilingEnd("isinf");
        }
        return $X;
    }

    /**
     * @return array<NDArray>
     */
    public function numericalGradient(
        ?float $h=null,
        ?callable $f=null,
        NDArray ...$variables) : array
    {
        if($h===null)
            $h = 1e-4;
        if(!is_callable($f)) {
            throw new InvalidArgumentException("f must callable or array of f and h");
        }
        $grads = [];
        $orgVariables = $variables;
        $variables = [];
        foreach ($orgVariables as $variable) {
            $variables[] = $this->copy($variable);
        }
        foreach($variables as $x) {
            $grad = $this->alloc($x->shape(),dtype:$x->dtype());
            $this->zeros($grad);
            $grads[] = $grad;
            $size = $x->size();
            $xx = $x->reshape([$x->size()]);
            //$idx = $x->offset();
            $gg = $grad->reshape([$grad->size()]);
            //$gidx = $grad->offset();
            $h2 = $h*2 ;
            for($i=0;$i<$size;$i++) {
                //    $value = $xx[$idx];
                $value = $this->copy($xx[R($i,$i+1)]);
                //    $xx[$idx] = $value + $h;
                $this->copy($this->increment($this->copy($value),$h),$xx[R($i,$i+1)]);
                //echo $value[0]."-h =>".$xx[$i]."\n";
                $y1 = $f(...$variables);
                //    $xx[$idx] = $value - $h;
                $this->copy($this->increment($this->copy($value),-$h),$xx[R($i,$i+1)]);
                //echo $value[0]."-h =>".$xx[$i]."\n";
                $y2 = $f(...$variables);
                $d = $this->axpy($y2,$this->copy($y1),-1);
                //    $gg[$gidx] = $this->sum($d)/$h2;
                $sum = $this->reduceSum($d->reshape([$d->size(),1]));
                //echo "d=".$sum[0]."\n";
                $this->copy($this->scal(1/$h2,$sum),$gg[R($i,$i+1)]);
                //    $xx[$idx] = $value;
                $this->copy($value,$xx[R($i,$i+1)]);
            }
        }
        return $grads;
    }

    public function isclose(NDArray $a, NDArray $b, float|object|null $rtol=null, ?float $atol=null,?bool $debug=null) : bool
    {
        if($a->dtype()!=$b->dtype()) {
            return false;
        }
        $isCpx = $this->isComplexDtype($a->dtype());
        if($rtol===null) {
            $rtol = $isCpx?C(1e-04):1e-04;
        }
        if($atol===null) {
            $atol = 1e-07;
        }
        if($a->shape()!=$b->shape()) {
            return false;
        }
        // diff = b - a
        $alpha =  $isCpx?C(-1):-1;
        $diffs = $this->axpy($a,$this->copy($b),$alpha);
        $diff = $this->abs($this->amax($diffs));

        // close = atol + rtol * b
        $scaled = $this->scal($rtol,$this->copy($b));
        $close = $atol+$this->abs($this->amax($scaled));

        if($debug) {
            echo "diff=".sprintf('%14.8e',$diff).", close=".sprintf('%14.8e',$close)."\n";
        }
        return $diff < $close;
    }


    public function solve(NDArray $a, NDArray $b, ?float $epsilon=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("solve");
        }
        if($epsilon===null) {
            $epsilon = 1e-7;
        }
        $aug = $this->augmentedMatrix($a,$b);
        [$m,$n] = $a->shape();
        if($m!=$n) {
            throw new InvalidArgumentException('matrix A must be square');
        }
        // forward
        for($i=0;$i<$n-1;$i++) {
            for($j=$i+1;$j<$n;$j++) {
                $div = $aug[$i][$i];
                if(abs($div)<$epsilon) {
                    throw new InvalidArgumentException('too small value');
                }
                $tmp = $aug[$j][$i] / $div;
                for($k=$i+1;$k<$n+1;$k++){
                    $aug[$j][$k] = $aug[$j][$k] - $tmp*$aug[$i][$k];
                }
            }
        }
        // backward
        for($i=$n-1;$i>=0;$i--) {
            for($j=$i+1;$j<$n;$j++) {
                $aug[$i][$n] = $aug[$i][$n] - $aug[$i][$j]*$aug[$j][$n];
            }
            $div = $aug[$i][$i];
            if(abs($div)<$epsilon) {
                throw new InvalidArgumentException('too small value');
            }
            $aug[$i][$n] = $aug[$i][$n] / $div;
        }
        $solve = $this->alloc([$n],dtype:$a->dtype());
        $SS = $solve->buffer();
        $this->blas->copy($n,$aug->buffer(),$aug->offset()+$n,$n+1,$SS,0,1);
        if($this->profiling) {
            $this->profilingEnd("solve");
        }
        return $solve;
    }

    public function solveg(NDArray $a, NDArray $b, ?float $epsilon=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("solveg");
        }
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
        $iscomplex = $this->cistype($a->dtype());
        $m = $shapeA[0];
        $n = $shapeA[1];
        for($j = 0; $j < $n; ++$j) {
            for($i = $j+1; $i < $m; ++$i) {
                $pa = $a[$j][[$j,$j+1]];
                $pb = $a[$i][[$j,$j+1]];
                if($iscomplex) {
                    $abs = $this->cabs($pb[0]);
                } else {
                    $abs = abs($pb[0]);
                }
                if($abs < $epsilon) {
                    continue;
                }
                [$r,$z,$c,$s] = $this->rotg($pa,$pb);

                $num_elements_A = $n - $j;
                $rowA_j_start = $a[$j][[$j,$j+$num_elements_A]]; // From the j-th column of the j-th row
                $rowA_i_start = $a[$i][[$j,$j+$num_elements_A]]; // From the j-th column of the i-th row
                $this->rot($rowA_j_start, $rowA_i_start, $c, $s);
                $this->rot($b[[$j,$j+1]], $b[[$i,$i+1]], $c, $s);
            }
        }

        $x = $this->alloc([$n],$a->dtype());

        //for ($i=$n-1; $i>=0; --$i) {
        //    if (abs($a[$i][$i]) < $epsilon) {
        //        throw new InvalidArgumentException("Matrix is singular or near-singular (zero pivot encountered at R[$i][$i]).");
        //    }
        //    $sum = 0;
        //    for ($k = $i+1; $k<$n; ++$k) {
        //        $sum += $a[$i][$k] * $x[$k];
        //    }
        //
        //    // x[i] = (b'[i] - sum) / R[i][i]
        //    $x[$i] = ($b[$i] - $sum) / $a[$i][$i];
        //}

        if($this->blas->hasIamin()) {
            $min_idx = $this->blas->iamin($n, $a->buffer(), $a->offset(), $n+1);
        } else {
            $min_idx = $this->blasIaminCompatible($n,$a->buffer(), $a->offset(), $n+1);
        }
        $min_diag = $a[$min_idx][$min_idx];
        if($iscomplex) {
            $min_diag_abs_value = $this->cabs($min_diag);
        } else {
            $min_diag_abs_value = abs($min_diag);
        }
        if ($min_diag_abs_value < $epsilon) {
            throw new RuntimeException("Matrix is singular or near-singular (minimum absolute diagonal $min_diag_abs_value at A[$min_idx][$min_idx] is below threshold).");
        }
        $this->copy($b[[0,$n]],$x);
        $this->trsv(
            $a,
            $x,
        );

        if($this->profiling) {
            $this->profilingEnd("solveg");
        }
        return $x;
    }
}
