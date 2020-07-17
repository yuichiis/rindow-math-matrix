<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;
use RuntimeException;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class PhpMath
{
    protected $math;
    protected $forceMath;
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];

    public function __construct($math=null,$forceMath=null)
    {
        $this->math = $math;
        $this->forceMath = $forceMath;
    }

    public function forceMath($forceMath)
    {
        $this->forceMath = $forceMath;
    }

    protected function useMath(Buffer $X)
    {
        if($this->math===null)
            return false;
        return $this->forceMath || in_array($X->dtype(),$this->floatTypes);
    }

    /**
     *     sum := sum(X)
     */
    public function sum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        if($this->math) {
            return $this->math->sum($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX;
        $acc = 0.0;
        for ($i=0; $i<$n; $i++,$idxX+=$incX) {
            $acc += $X[$idxX];
        }
        return $acc;
    }

    /**
     *     index := max(X)
     */
    public function imax(
        int $n,
        Buffer $X, int $offsetX, int $incX) : int
    {
        if($this->useMath($X)) {
            return $this->math->imax($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX+$incX;
        $acc = $X[$offsetX];
        $idx = 0;
        for($i=1; $i<$n; $i++,$idxX+=$incX) {
            if($acc < $X[$idxX]) {
                $acc = $X[$idxX];
                $idx = $i;
            }
        }
        return $idx;
    }

    /**
     *     index := min(X)
     */
    public function imin(
        int $n,
        Buffer $X, int $offsetX, int $incX) : int
    {
        if($this->useMath($X)) {
            return $this->math->imin($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX+$incX;
        $acc = $X[$offsetX];
        $idx = 0;
        for($i=1; $i<$n; $i++,$idxX+=$incX) {
            if($acc > $X[$idxX]) {
                $acc = $X[$idxX];
                $idx = $i;
            }
        }
        return $idx;
    }

    /**
     *     X := a*X + b
     */
    public function increment(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        if($this->useMath($X)) {
            $this->math->increment($n,$alpha,$X,$offsetX,$incX,$beta);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = $alpha*$X[$idx] + $beta;
        }
    }

    /**
     *     X := 1 / (a*X + b)
     */
    public function reciprocal(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        if($this->useMath($X)) {
            $this->math->reciprocal($n,$alpha,$X,$offsetX,$incX,$beta);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $alpha*$X[$idx] + $beta;
            if($t==0.0) {
                throw new RuntimeException('Zero divide.');
            }
            $X[$idx] = 1 / $t;
        }
    }

    /**
     *     X := X  (X > a)
     *     X := a  (X <= a)
     */
    public function maximum(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->maximum($n,$alpha,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            if($X[$idx] < $alpha) {
                $X[$idx] = $alpha;
            }
        }
    }

    /**
     *     X := X  (X < a)
     *     X := a  (X >= a)
     */
    public function minimum(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->minimum($n,$alpha,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            if($X[$idx] > $alpha) {
                $X[$idx] = $alpha;
            }
        }
    }

    /**
     *     X := 1  (X > a)
     *     X := 0  (X <= a)
     */
    public function greater(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->greater($n,$alpha,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            if($X[$idx] > $alpha) {
                $X[$idx] = 1.0;
            } else {
                $X[$idx] = 0.0;
            }
        }
    }

    /**
     *     X := 1  (X < a)
     *     X := 0  (X >= a)
     */
    public function less(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->less($n,$alpha,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            if($X[$idx] < $alpha) {
                $X[$idx] = 1.0;
            } else {
                $X[$idx] = 0.0;
            }
        }
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
    public function multiply(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->multiply($trans,$m,$n,$X,$offsetX,$incX,$A,$offsetA,$ldA);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idAj = $offsetA;
        for($j=0; $j<$rows; $j++,$idAj+=$incAj) {
            $idA = $idAj;
            $idX = $offsetX;
            for($i=0; $i<$cols; $i++,$idA+=$incAi,$idX+=$incX) {
                $A[$idA] = $X[$idX] * $A[$idA];
            }
        }
    }

    /**
     *     Y(m,n) := alpha * X(n) + Y(m,n)
     */
    public function add(
        bool $trans,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->add($trans,$m,$n,$alpha,$X,$offsetX,$incX,$A,$offsetA,$ldA);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idAj = $offsetA;
        for($j=0; $j<$rows; $j++,$idAj+=$incAj) {
            $idA = $idAj;
            $idX = $offsetX;
            for($i=0; $i<$cols; $i++,$idA+=$incAi,$idX+=$incX) {
                $A[$idA] = $alpha * $X[$idX] + $A[$idA];
            }
        }
    }

    /**
     *     X := X ^ 2
     */
    public function square(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->square($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = $t * $t;
        }
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->sqrt($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            if($t<0.0) {
                throw new RuntimeException('Invalid value in sqrt.');
            }
            $X[$idx] = sqrt($t);
        }
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        if($this->useMath($X)) {
            $this->math->rsqrt($n,$alpha,$X,$offsetX,$incX,$beta);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            if($t<0.0) {
                throw new RuntimeException('Invalid value in sqrt.');
            }
            $t = $alpha*sqrt($t) + $beta;
            if($t==0.0) {
                throw new RuntimeException('Zero divide.');
            }
            $X[$idx] = 1 / $t;
        }
    }

    /**
     *     X := X ^ a
     */
    public function pow(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->pow($n,$alpha,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = $X[$idx] ** $alpha;
        }
    }

    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->exp($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = exp($X[$idx]);
        }
    }

    /**
     *     X := log(X)
     */
    public function log(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->log($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            if($t<0.0) {
                throw new RuntimeException('Invalid value in log.');
            }
            $X[$idx] = log($t);
        }
    }

    /**
     *     A(m,n) := X(n)
     */
    public function duplicate(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->duplicate($trans,$m,$n,$X,$offsetX,$incX,$A,$offsetA,$ldA);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idA = $offsetA;
        for($j=0; $j<$rows; $j++) {
            $this->duplicate_blas_copy($cols,$X,$offsetX,$incX,$A,$idA,$incAi);
            $idA += $incAj;
        }
    }
    protected function duplicate_blas_copy($n,$X,$offsetX,$incX,$Y,$offsetY,$incY)
    {
        $idX = $offsetX;
        $idY = $offsetY;
        for($i=0; $i<$n; $i++,$idX+=$incX,$idY+=$incY) {
            $Y[$idY] = $X[$idX];
        }
    }

    /**
     *     X := 0
     */
    public function zeros(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->zeros($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = 0;
        }
    }

    /**
     *     Y := A(k,X(m))
     */
    public function selectAxis0(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $ldY
        ) : void
    {
        if($this->math) {
            $this->math->selectAxis0($m,$n,$k,$A,$offsetA,$ldA,$X,$offsetX,$incX,$Y,$offsetY,$ldY);
            return;
        }

        if($offsetA+($m-1)*$ldA+$n-1>=count($A))
            throw new RuntimeException('Vector specification too large for bufferA.');
        if($offsetX+($k-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for bufferX.');
        if($offsetY+($k-1)*$ldY+$n-1>=count($Y))
            throw new RuntimeException('Vector specification too large for bufferY.');

        $idx = $offsetX;
        $idy = $offsetY;
        for($i=0; $i<$k; $i++,$idx+=$incX,$idy+=$ldY) {
            $label = (int)$X[$idx];
            if($label>=$m||$label<0)
                throw new RuntimeException('Label number is out of bounds.');
            $idA = $offsetA+$ldA*$label;
            if($n==1) {
                $Y[$idy]  = $A[$idA];
            } else {
                $this->rindow_openblas_math_copy($n, $A,$idA,1, $Y,$idy,1);
            }
        }
    }

    protected function rindow_openblas_math_copy($n,$X,$offsetX,$incX,$Y,$offsetY,$incY)
    {
        $idX = $offsetX;
        $idY = $offsetY;
        for($i=0; $i<$n; $i++, $idX+=$incX,$idY+=$incY) {
            $Y[$idY] = $X[$idX];
        }
    }

    /**
     *     Y := A(k,X(m))
     */
    public function selectAxis1(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->selectAxis1($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if($offsetA+($m-1)*$ldA+$n-1>=count($A))
            throw new RuntimeException('Vector specification too large for bufferA.');
        if($offsetX+($m-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for bufferX.');
        if($offsetY+($m-1)*$incY>=count($Y))
            throw new RuntimeException('Vector specification too large for bufferY.');

        $ida = $offsetA;
        $idx = $offsetX;
        $idy = $offsetY;
        for ($i=0; $i<$m; $i++,$ida+=$ldA,$idx+=$incX,$idy+=$incY) {
            $label = (int)$X[$idx];
            if($label>=$n||$label<0)
                throw new RuntimeException('Label number is out of bounds.');
            $Y[$idy] = $A[$ida+$label];
        }
    }

    /**
     *     Y := updateAddOnehot(X,a)
     */
    public function updateAddOnehot(
        int $m,
        int $n,
        float $a,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $ldY
        ) : void
    {
        if($this->useMath($Y)) {
            $this->math->updateAddOnehot($m,$n,$a,$X,$offsetX,$incX,$Y,$offsetY,$ldY);
            return;
        }

        if($offsetX+($m-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for bufferX.');
        if($offsetY+($m-1)*$ldY+($n-1)>=count($Y))
            throw new RuntimeException('Vector specification too large for bufferY.');

        $idx = $offsetX;
        $idy = $offsetY;
        for ($i=0; $i<$m; $i++,$idy+=$ldY,$idx+=$incX) {
            $label = (int)$X[$idx];
            if($label>=$n||$label<0)
                throw new RuntimeException('Label number is out of bounds.');
            $Y[$idy+$label] = $Y[$idy+$label] + $a;
        }
    }

    /**
     * Y(i) := 1  ( X(i) == Y(i) )
     * Y(i) := 0  ( X(i) != Y(i) )
     */
    public function equal(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        if($this->math) { // Support all dtype by math
            $this->math->equal($n,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetY+($n-1)*$incY>=count($Y))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idX = $offsetX;
        $idY = $offsetY;
        if(is_bool($Y[$idY])) {$true = true; $false = false;}
        else                   {$true = 1; $false = 0;}
        for($i=0; $i<$n; $i++,$idX+=$incX,$idY+=$incY) {
            $Y[$idY] = ($Y[$idY] == $X[$idX]) ? $true : $false;
        }
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->reduceSum($trans,$m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetX+($rows-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idAj = $offsetA;
        $idX = $offsetX;
        for($j=0; $j<$rows; $j++,$idAj+=$incAj,$idX+=$incX) {
            $sum = 0;
            $idA = $idAj;
            for($i=0; $i<$cols; $i++,$idA+=$incAi) {
                $sum += $A[$idA];
            }
            $X[$idX] = $sum;
        }
    }
    
    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceMax(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->reduceMax($trans,$m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetX+($rows-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idAj = $offsetA;
        $idX = $offsetX;
        for($j=0; $j<$rows; $j++,$idAj+=$incAj,$idX+=$incX) {
            $idA = $idAj;
            $max = $A[$idA];
            for($i=1; $i<$cols; $i++,$idA+=$incAi) {
                $na = $A[$idA];
                if($max<$na)
                    $max = $na;
            }
            $X[$idX] = $max;
        }
    }

    public function softmax(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA) : void
    {
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idA = $offsetA;
        for($i=0;$i<$m;$i++,$idA+=$ldA) {
            //float t,max_a,sum_exp;
            $max_a = $this->softmax_max($n,$A,$idA,1);
            $sum_exp = 0;
            for($j=0;$j<$n;$j++) {
                $t = exp($A[$idA+$j]-$max_a);
                $sum_exp += $t;
                $A[$idA+$j] = $t;
            }
            if($sum_exp==0.0) {
                throw new RuntimeException("Zero divide in softmax.");
            }
            for($j=0;$j<$n;$j++) {
                $A[$idA+$j] = $A[$idA+$j] / $sum_exp;
            }
        }
    }

    protected function softmax_max($n,$x,$offsetX,$incX)
    {
        $a = $x[$offsetX];
        $idX = $offsetX+$incX;
        for($i=1;$i<$n;$i++,$idX+=$incX) {
            if($a<$x[$idX]) {
                $a = $x[$idX];
            }
        }
        return $a;
    }

    public function astype(
        int $n,
        int $dtype,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        if($this->math) {
            $this->math->astype($n,$dtype,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if(in_array($dtype,$this->floatTypes)) {
            $isFloat = true;
        } elseif(in_array($dtype,$this->intTypes)) {
            $isFloat = false;
            if($dtype==NDArray::uint8) {
                $mask = 0xff;
            } elseif($dtype==NDArray::uint16) {
                $mask = 0xffff;
            } elseif($dtype==NDArray::uint32) {
                $mask = 0xffffffff;
            } else {
                $mask = null;
            }
        } elseif($dtype==NDArray::bool) {
            $isFloat = false;
        } else {
            throw new InvalidArgumentException('dtype must be type of integer or float: '.$dtype);
        }
        if(is_bool($X[0])) {
            $fromBoolean = true;
        } else {
            $fromBoolean = false;
        }

        $idx = $offsetX;
        $idy = $offsetY;
        if($fromBoolean) {
            if($isFloat) {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = ($X[$idx]) ? 1.0 : 0.0;
                }
            } else {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = ($X[$idx]) ? 1 : 0;
                }
            }
        } else {
            if($dtype==NDArray::bool) {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = ($X[$idx]) ? true : false;
                }
            } elseif($isFloat) {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = (float)($X[$idx]);
                }
            } else {
                if($mask) {
                    for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                        $Y[$idy] = (int)($mask & $X[$idx]);
                    }
                } else {
                    for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                        $Y[$idy] = (int)($X[$idx]);
                    }
                }
            }
        }
    }
    
    /**
     * copy a image with channels
     */
    protected function copyCell2d(
        bool $reverse,
        Buffer $images,
        int $images_pos,
        int $filter_h,
        int $filter_w,
        int $channels,
        int $channel_step,
        int $filter_w_step,
        int $filter_h_step,
        int $vin_y,
        int $vin_x,
        int $vin_h,
        int $vin_w,
        int $out,
        int $out_pos,
        int $out_filter_step,
        int $out_channel_step
        )
    {
        #print('v=%d,%d,%d,%d' % (vin_y,vin_x,vin_h,vin_w))
        $filter_h_pos = $images_pos;
        $out_filter_pos = $out_pos;
        for($y=0; $y<$filter_h; $y++) {
            $yy = $y+$vin_y;
            $filter_w_pos = $filter_h_pos;
            for($x=0; $x<$filter_w; $x++) {
                $channel_pos = $filter_w_pos;
                $out_channel_pos = $out_filter_pos;
                $xx = $x+$vin_x;
                #print('yx=%d,%d' % (yy,xx))
                for($c=0; $c<$channels; $c++) {
                    if($yy<0 || $yy>=$vin_h ||
                       $xx<0 || $xx>=$vin_w) {
                        #print('pad') 
                        if(!$reverse) {
                            $out[$out_channel_pos] = 0;
                        }
                    } else {
                        if(!$reverse) {
                            $out[$out_channel_pos] =  $images[$channel_pos];
                        } else {
                             $images[$channel_pos] = $out[$out_channel_pos];
                        }
                    }
                    $out_channel_pos = $out_filter_pos;
                    $channel_pos += $channel_step;
                }
                $out_filter_pos += $out_filter_step;
                $filter_w_pos += $filter_w_step;
            }
            $filter_h_pos += $filter_h_step;
        }
        return $out_pos;
    }
    
    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col2d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $in_h,
        int $in_w,
        int $channels,
        int $filter_h,
        int $filter_w,
        int $strides_h,
        int $strides_w,
        bool $padding,
        bool $channels_first,
        bool $cols_channels_first,
        Buffer $out,
        int $out_offset,
        int $out_size
        )
    {
        $images_buf_size = $batches*$in_h*$in_w*$channels;
        if($images_size!=$images_buf_size ||
            count($images)-$images_offset<$images_buf_size) {
            throw new InvalidArgumentException('images buffer size is invalid');
        }
        $out_h = floor(($in_h-$filter_h)/$stride_h)+1;
        $out_w = floor(($in_w-$filter_w)/$stride_w)+1;
        if($padding) {
            $out_buf_size = 
                $batches*
                $in_h*$filter_h*
                $in_w*$filter_w*
                $channels;
            #print('outsz=',out.shape)
            $start_h = -floor(($in_h-$out_h)/2)
            $start_w = -floor(($in_w-$out_w)/2);
            $end_h = $start_h+$in_h;
            $end_w = $start_w+$in_w;
            #print('start-end=(%d,%d)-(%d,%d)'%(start_h,start_w,end_h,end_w))
        } else {
            $start_h = $start_w = 0;
            $end_h = $out_h;
            $end_w = $out_w;
            $out_buf_size = $batches*
                $out_h*$filter_h*
                $out_w*$filter_w*
                $channels;
        }
        if($out_size!=$out_buff_size ||
            count($out)-$out_offset>$out_buff_size) {
            throw new InvalidArgumentException('output buffer size is invalid');
        }
        if($channels_first) {
            # stride parameters
            $stride_w_step = $stride_w;
            $stride_h_step = $in_w*$stride_h;
            $batch_step = $channels*$in_w*$in_h;
            # copy parameters
            $channel_step = $in_h*$in_w;
            $filter_w_step = 1;
            $filter_h_step = $in_w;
        } else {
            # stride parameters
            $stride_w_step = $channels*$stride_w;
            $stride_h_step = $channels*$in_w*$stride_h;
            $batch_step = $channels*$in_w*$in_h;
            # copy parameters
            $channel_step = 1;
            $filter_w_step = $channels;
            $filter_h_step = $filter_w_step*$in_w;
        }
        if($cols_data_format == 'channels_first') {
            $out_filter_step = 1;
            $out_channel_step = $filter_h*$filter_w;
        } else {
            $out_filter_step = $channels;
            $out_channel_step = 1;
        }
        $out_cell_step = $filter_h*$filter_w*$channels;
        
        $out_pos = $out_offset;
        $batch_pos = 0;
    
        $start_vin_y = $start_h*$stride_h;
        $start_vin_x = $start_w*$stride_w;
        $vin_h = ($out_h-1)*$stride_h+$filter_h;
        $vin_w = ($out_w-1)*$stride_w+$filter_w;
    
        for($batch=0; $batch<$batches;$batch++) {
            $stride_h_pos = $batch_pos+($start_h*$stride_h_step);
            $vin_y = $start_vin_y;
            for ($y=$start_h;$y<$end_h;$y++){
                $stride_w_pos = $stride_h_pos+($start_w*$stride_w_step);
                $vin_x = $start_vin_x;
                for($x=$start_w;$x<$end_w;$x++) {
                    #print('osf=%d,%d,%d'%(out_h,stride_h,filter_h))
                    $out_pos = $this->copyCell2d(
                        $reverse,
                        $images,
                        $stride_w_pos,
                        $filter_h,
                        $filter_w,
                        $channels,
                        $channel_step,
                        $filter_w_step,
                        $filter_h_step,
                        $vin_y,
                        $vin_x,
                        $vin_h,
                        $vin_w,
                        $out,
                        $out_pos,
                        $out_filter_step,
                        $out_channel_step
                    );
                    $stride_w_pos += $stride_w_step;
                    $vin_x += $stride_w;
                    $out_pos += $out_cell_step;
                }
                $stride_h_pos += $stride_h_step;
                $vin_y += $stride_h;
            }    
            $batch_pos += $batch_step;
        }
        return $out;
    }
}
