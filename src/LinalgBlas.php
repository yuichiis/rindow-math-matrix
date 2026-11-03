<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\Buffer;
use InvalidArgumentException;

trait LinalgBlas
{
    /**
    *    Y := X
    */
    public function copy(
        NDArray $X,
        ?NDArray $Y=null ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("copy");
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),dtype:$X->dtype());
        } else {
            if($X->shape()!=$Y->shape()) {
                $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }
        $dtype = $X->dtype();
        if($dtype!=$Y->dtype()) {
            throw new InvalidArgumentException(
                "Unmatch dtype of dimension: ".
                $this->dtypeToString($dtype).','.
                $this->dtypeToString($Y->dtype())
            );
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        if($dtype===NDArray::float32 &&
           $dtype===NDArray::float64 &&
           $dtype===NDArray::complex64 &&
           $dtype===NDArray::complex128
           ) {
            $this->blas->copy($N,$XX,$offX,1,$YY,$offY,1);
        } else {
            $this->math->copyBuffer($N,$XX,$offX,$YY,$offY);
        }
        if($this->profiling) {
            $this->profilingEnd("copy");
        }
        return $Y;
    }

    /**
    *    X := alpha * X
    */
    public function scal(
        float|object $alpha,
        NDArray $X) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scal");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->blas->scal($N,$alpha,$XX,$offX,1);
        if($this->profiling) {
            $this->profilingEnd("scal");
        }
        return $X;
    }

    /**
    *    Y := alpha * X + Y
    */
    public function axpy(
        NDArray $X,
        NDArray $Y,
        float|object|null $alpha=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("axpy");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$X->dtype());
        }
        $this->blas->axpy($N,$alpha,$XX,$offX,1,$YY,$offY,1);
        if($this->profiling) {
            $this->profilingEnd("axpy");
        }
        return $Y;
    }

    /**
    *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
    */
    public function dot(
        NDArray $X,
        NDArray $Y,
        ?bool $conj=null
        ) : float|object
    {
        if($this->profiling) {
            $this->profilingStart("dot");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        if(!$this->isComplex($X->dtype())) {
            $result = $this->blas->dot($N,$XX,$offX,1,$YY,$offY,1);
        } elseif($conj===false) { // explicit conjugation false
            $result = $this->crebuild($this->blas->dotu($N,$XX,$offX,1,$YY,$offY,1));
        } else {            // implicit conjugation true
            $result = $this->crebuild($this->blas->dotc($N,$XX,$offX,1,$YY,$offY,1));
        }
        if($this->profiling) {
            $this->profilingEnd("dot");
        }
        return $result;
    }

    /**
    *    ret := |x_1| + ... + |x_n|
    */
    public function asum(
        NDArray $X) : float
    {
        if($this->profiling) {
            $this->profilingStart("asum");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $result = $this->blas->asum($N,$XX,$offX,1);
        if($this->profiling) {
            $this->profilingEnd("asum");
        }
        return $result;
    }

    /**
    *    ret := arg max |X(i)|
    */
    public function iamax(
        NDArray $X) : int
    {
        if($this->profiling) {
            $this->profilingStart("iamax");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $result = $this->blas->iamax($N,$XX,$offX,1);
        if($this->profiling) {
            $this->profilingEnd("iamax");
        }
        return $result;
    }

    /**
    *    ret := arg min |X(i)|
    */
    public function iamin(
        NDArray $X) : int
    {
        if($this->profiling) {
            $this->profilingStart("iamin");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($this->blas->hasIamin()) {
            $result = $this->blas->iamin($N,$XX,$offX,1);
        } else {
            $result = $this->blasIaminCompatible($N,$XX,$offX,1);
        }
        if($this->profiling) {
            $this->profilingEnd("iamin");
        }
        return $result;
    }

    /**
    *   legacy opennblas compatible
    */
    public function blasIaminCompatible(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        if($this->iaminwarning) {
            echo "*iamin* not found. probably OpenBLAS is legacy version.";
            $this->iaminwarning = true;
        }
        if($offsetX+($n-1)*$incX>=count($X)) {
            throw new InvalidArgumentException('Vector X specification too large for buffer.');
        }
        $idxX = $offsetX+$incX;
        if($this->isComplex($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            $idx = 0;
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                if($acc > $this->cabs($X[$idxX])) {
                    $acc = $this->cabs($X[$idxX]);
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            $idx = 0;
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                if($acc > abs($X[$idxX])) {
                    $acc = abs($X[$idxX]);
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    /**
    *    ret := sqrt(sum(Xn ** 2))
    */
    public function nrm2(
        NDArray $X) : float
    {
        if($this->profiling) {
            $this->profilingStart("nrm2");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $ret = $this->blas->nrm2($N,$XX,$offX,1);
        if($this->profiling) {
            $this->profilingEnd("nrm2");
        }
        return $ret;
    }

    /**
     * a,b,c,s = rotg(x,y)
     * @return array<NDArray>
     */
    public function rotg(
        NDArray $X,
        NDArray $Y,
        ?NDArray $R=null,
        ?NDArray $Z=null,
        ?NDArray $C=null,
        ?NDArray $S=null) : array
    {
        if($this->profiling) {
            $this->profilingStart("rotg");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $R = $this->copy($X,$R);
        $Z = $this->copy($Y,$Z);
        if($C==null) {
            $C = $this->alloc($X->shape(),dtype:$X->dtype());
        }
        if($S==null) {
            $S = $this->alloc($Y->shape(),dtype:$X->dtype());
        }
        $AA = $R->buffer();
        $offA = $R->offset();
        $BB = $Z->buffer();
        $offB = $Z->offset();
        $CC = $C->buffer();
        $offC = $C->offset();
        $SS = $S->buffer();
        $offS = $S->offset();
        $this->blas->rotg(
            $AA,$offA,
            $BB,$offB,
            $CC,$offC,
            $SS,$offS
        );
        if($this->profiling) {
            $this->profilingEnd("rotg");
        }
        return [$R,$Z,$C,$S];
    }

    /**
    *    x,y := rot(x,y,c,s)
    */
    public function rot(
        NDArray $X,
        NDArray $Y,
        NDArray $C,
        NDArray $S) : void
    {
        if($this->profiling) {
            $this->profilingStart("rot");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $CC = $C->buffer();
        $offC = $C->offset();
        $SS = $S->buffer();
        $offS = $S->offset();
        $this->blas->rot($N,
            $XX,$offX,1,$YY,$offY,1,
            $CC,$offC,$SS,$offS
        );
        if($this->profiling) {
            $this->profilingEnd("rot");
        }
    }

    /**
     * d1,d2,b1,p = rotmg(x,y)   b1: rotated x   p: params  d1,d2:works
     * @return array<NDArray>
     */
    public function rotmg(
        NDArray $X,
        NDArray $Y,
        ?NDArray $D1=null,
        ?NDArray $D2=null,
        ?NDArray $B1=null,
        ?NDArray $P=null,
        ) : array
    {
        if($this->profiling) {
            $this->profilingStart("rotmg");
        }
        if($X->size()!=1||$Y->size()!=1) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($D1==null) {
            $D1 = $this->ones($this->alloc([],dtype:$X->dtype()));
        }
        if($D2==null) {
            $D2 = $this->ones($this->alloc([],dtype:$X->dtype()));
        }
        if($B1==null) {
            $B1 = $this->alloc([],dtype:$X->dtype());
        }
        if($P==null) {
            $P = $this->zeros($this->alloc([5],dtype:$X->dtype()));
        }
        $this->copy($X->reshape([1]),$B1->reshape([1]));

        $DD1 = $D1->buffer();
        $offD1 = $D1->offset();
        $DD2 = $D2->buffer();
        $offD2 = $D2->offset();
        $BB1 = $B1->buffer();
        $offB1 = $B1->offset();
        $BB2 = $Y->buffer();
        $offB2 = $Y->offset();
        $PP = $P->buffer();
        $offP = $P->offset();
        $this->blas->rotmg(
            $DD1,$offD1,
            $DD2,$offD2,
            $BB1,$offB1,
            $BB2,$offB2,
            $PP,$offP
        );
        if($this->profiling) {
            $this->profilingEnd("rotmg");
        }
        return [$D1,$D2,$B1,$P];
    }

    /**
    *    x,y := rot(x,y,p)
    */
    public function rotm(
        NDArray $X,
        NDArray $Y,
        NDArray $P,
        ) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotm");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $PP = $P->buffer();
        $offP = $P->offset();
        $this->blas->rotm(
            $N,
            $XX,$offX,1,
            $YY,$offY,1,
            $PP,$offP,
        );
        if($this->profiling) {
            $this->profilingEnd("rotm");
        }
    }

    /**
    *    Y := X
    *    X := Y
    */
    public function swap(
        NDArray $X,
        NDArray $Y) : void
    {
        if($this->profiling) {
            $this->profilingStart("swap");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = $this->shapeToString($X->shape()).','.$this->shapeToString($Y->shape());
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $this->blas->swap($N,$XX,$offX,1,$YY,$offY,1);
        if($this->profiling) {
            $this->profilingEnd("swap");
        }
    }

    /**
    *    y := alpha * Ax + beta * y
    */
    public function gemv(
        NDArray $A,
        NDArray $X,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $Y=null,
        ?bool $trans=null,
        ?bool $conj=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gemv");
        }
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $X->ndim()!=1) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray and "X" must 1D-NDArray.');
        }
        $shapeA = $A->shape();
        $shapeX = $X->shape();
        $rows = (!$trans) ? $shapeA[0] : $shapeA[1];
        $cols = (!$trans) ? $shapeA[1] : $shapeA[0];
        if($cols!=$shapeX[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and The number of item in "X" must be the same');
        }
        $AA = $A->buffer();
        $XX = $X->buffer();
        $offA = $A->offset();
        $offX = $X->offset();
        $m = $shapeA[0];
        $n = $shapeA[1];
        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($Y!=null) {
            if($Y->ndim()!=1) {
                throw new InvalidArgumentException('"Y" must 1D-NDArray.');
            }
            $shapeY = $Y->shape();
            if($rows!=$shapeY[0]) {
                throw new InvalidArgumentException('The number of rows in "A" and The number of item in "Y" must be the same');
            }
        } else {
            $Y = $this->zeros($this->alloc([$rows],dtype:$X->dtype()));
        }
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $trans = $this->transToCode($trans,$conj);

        $this->blas->gemv(
            BLAS::RowMajor,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$n,
            $XX,$offX,1,
            $beta,
            $YY,$offY,1);

        if($this->profiling) {
            $this->profilingEnd("gemv");
        }
        return $Y;
    }

    /**
    *    Ax = b (triangle solve)
    */
    public function trsv(
        NDArray $A,
        NDArray $X,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ?bool $unit=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("trsv");
        }
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $X->ndim()!=1) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray and "X" must 1D-NDArray.');
        }
        $shapeA = $A->shape();
        $shapeX = $X->shape();
        if($shapeA[0]!=$shapeA[1]) {
            throw new InvalidArgumentException('Matrix A must be square.: '.$this->shapeToString($shapeA).' given.');
        }
        if($shapeA[0]!=$shapeX[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and The number of item in "X" must be the same.');
        }
        $AA = $A->buffer();
        $XX = $X->buffer();
        $offA = $A->offset();
        $offX = $X->offset();
        $n = $shapeA[0];
        $trans = $this->transToCode($trans,$conj);
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;

        $this->blas->trsv(
            BLAS::RowMajor,
            $uplo,
            $trans,
            $diag,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
        );

        if($this->profiling) {
            $this->profilingEnd("trsv");
        }
        return $X;
    }

    /**
    *    C := alpha * AB + beta * C
    */
    public function gemm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $transA=null,
        ?bool $transB=null,
        ?bool $conjA=null,
        ?bool $conjB=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gemm");
        }
        [$transA,$conjA] = $this->complementTrans($transA,$conjA,$A->dtype());
        [$transB,$conjB] = $this->complementTrans($transB,$conjB,$B->dtype());
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        if($transA) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $shapeB = $B->shape();
        if($transB) {
            $shapeB = [$shapeB[1],$shapeB[0]];
        }
        if($shapeA[1]!=$shapeB[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and the number of rows in "B" must be the same');
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();
        $M = $shapeA[0];
        $N = $shapeB[1];
        $K = $shapeA[1];

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
            }
        } else {
            $C = $this->zeros($this->alloc([$M,$N],dtype:$A->dtype()));
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = $this->transToCode($transA,$conjA);
        $transB = $this->transToCode($transB,$conjB);

        $this->blas->gemm(
            BLAS::RowMajor,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        if($this->profiling) {
            $this->profilingEnd("gemm");
        }
        return $C;
    }

    /**
    *    A = Symmetric matrix
    *    C := alpha * AB + beta * C  ( right = false )
    *    C := alpha * BA + beta * C  ( right = true )
    */
    public function symm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $right=null,
        ?bool $lower=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("symm");
        }
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $rowsA = $shapeA[0];
        if($rowsA!=$shapeA[1]) {
            throw new InvalidArgumentException('The matrix "A" must be symmetric');
        }
        $shapeB = $B->shape();
        $M = $shapeB[0];
        $N = $shapeB[1];
        $tmpB = ($right) ? $N : $M;
        if($rowsA!=$tmpB) {
            throw new InvalidArgumentException('Unmatch Shape of matrix "A" and "B": '."($rowsA,$rowsA) != ($M,$N)");
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('Matrix "B" and "C" must be same shape');
            }
        } else {
            $C = $this->zeros($this->alloc([$M,$N],dtype:$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = $rowsA;
        $ldb = $N;
        $ldc = $N;
        $side = ($right) ? BLAS::Right : BLAS::Left;
        $uplo = ($lower) ? BLAS::Lower : BLAS::Upper;

        $this->blas->symm(
            BLAS::RowMajor,$side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        if($this->profiling) {
            $this->profilingEnd("symm");
        }
        return $C;
    }

    /**
    *    C := alpha * A A^T + beta * C  (trans=false)
    *    C := alpha * A^T A + beta * C  (trans=true)
    */
    public function syrk(
        NDArray $A,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("syrk");
        }
        $trans = $trans ?? false;
        // $conj = $conj ?? $trans; // Doing so will result in an error.
        $conj = false;  // conj must be false

        if($A->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        if($trans) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $N = $shapeA[0];
        $K = $shapeA[1];

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->zeros($this->alloc([$N,$N],dtype:$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->syrk(
            BLAS::RowMajor,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc);

        if($this->profiling) {
            $this->profilingEnd("syrk");
        }
        return $C;
    }

    /**
    *    C := alpha * A B^T + alpha * B A^T + beta * C  (trans=false)
    *    C := alpha * B A^T + alpha * A B^T + beta * C  (trans=true)
    */
    public function syr2k(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("syr2k");
        }
        $trans = $trans ?? false;
        // $conj = $conj ?? $trans; // Doing so will result in an error.
        $conj = false;  // conj must be false

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($shapeA!=$shapeB) {
            throw new InvalidArgumentException('Matrix A and B must be same shape');
        }
        if($trans) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $N = $shapeA[0];
        $K = $shapeA[1];

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->zeros($this->alloc([$N,$N],dtype:$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldb = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->syr2k(
            BLAS::RowMajor,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        if($this->profiling) {
            $this->profilingEnd("syr2k");
        }
        return $C;
    }

    /**
    *    B(m,n) := alpha * A(m,m) B(m,n)  (right=false)
    *    B(m,n) := alpha * B(m,n) A(n,n)  (right=true)
    */
    public function trmm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        ?bool $right=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ?bool $unit=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("trmm");
        }
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($shapeA[0]!=$shapeA[1]) {
            throw new InvalidArgumentException('Matrix A must be square.: '.
                '['.implode(',',$shapeA).']');
        }
        if($right) {
            $sizeA = $shapeB[1];
        } else {
            $sizeA = $shapeB[0];
        }
        if($sizeA!=$shapeA[0]) {
            throw new InvalidArgumentException('Unmatch shape of Matrix A and B: '.
                '['.implode(',',$shapeA).'] <=> ['.implode(',',$shapeB).']');
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $M = $shapeB[0];
        $N = $shapeB[1];

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->trmm(
            BLAS::RowMajor,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb);

        if($this->profiling) {
            $this->profilingEnd("trmm");
        }
        return $B;
    }

    /**
    *    C := alpha A^-1 B  (right=false)
    *    C := alpha B A^-1  (right=true)
    */
    public function trsm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        ?bool $right=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ?bool $unit=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("trsm");
        }
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($right) {
            $sizeA = $shapeB[1];
        } else {
            $sizeA = $shapeB[0];
        }
        if($sizeA!=$shapeA[0]) {
            throw new InvalidArgumentException('Unmatch shape of Matrix A and B: '.
                '['.implode(',',$shapeA).'] <=> ['.implode(',',$shapeA).']');
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $M = $shapeB[0];
        $N = $shapeB[1];

        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->trsm(
            BLAS::RowMajor,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb);

        if($this->profiling) {
            $this->profilingEnd("trsm");
        }
        return $B;
    }

    /**
     *  B := A    (trans=false)
     *  B := A^T  (trans=true)
     */
    public function omatcopy(
        NDArray $A,
        ?bool $trans=null,
        ?bool $conj=null,
        float|object|null $alpha=null,
        ?NDArray $B=null,
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("omatcopy");
        }
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        [$rows,$cols] = $A->shape();
        if($trans) {
            [$rows,$cols] = [$cols,$rows];
        }
        if($B===null) {
            $B = $this->zeros($this->alloc([$rows,$cols],dtype:$A->dtype()));
        } else {
            if($B->shape()!=[$rows,$cols]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        [$M,$N] = $A->shape();
        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $N;
        $BB = $B->buffer();
        $offB = $B->offset();
        $ldB = $cols;

        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        if($this->blas->hasOmatcopy()) {
            $this->blas->omatcopy(
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            );
        } else {
            $this->math->omatcopy(
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            );
        }

        if($this->profiling) {
            $this->profilingEnd("omatcopy");
        }
        return $B;
    }

}