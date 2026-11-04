<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use LogicException;
use InvalidArgumentException;
use RuntimeException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;
use Rindow\Math\Matrix\ComplexUtils;

class PhpBlas
{
    use Utils;
    use ComplexUtils;

    const GAM = 4096;
    const GAMSQ = 16777216;
    const RGAMSQ = 5.9604645e-8;

    protected ?object $blas;
    protected ?bool $forceBlas;
    /** @var array<int> $floatTypes */
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];
    protected bool $lapackCompatible = true;

    public function __construct(?object $blas=null,?bool $forceBlas=null)
    {
        //$this->blas = $blas;
        //$this->forceBlas = $forceBlas;
        $this->blas = null;
        $this->forceBlas = null;
    }

    //public function forceBlas($forceBlas)
    //{
    //    $this->forceBlas = $forceBlas;
    //}

    //protected function useBlas(Buffer $X)
    //{
    //    //if($this->blas===null)
    //    //    return false;
    //    //return $this->forceBlas || in_array($X->dtype(),$this->floatTypes);
    //    return false;
    //}

    public function getNumThreads() : int
    {
        if($this->blas===null)
            return 1;
        return $this->blas->getNumThreads();
    }

    public function getNumProcs() : int
    {
        if($this->blas===null)
            return 1;
        return $this->blas->getNumProcs();
    }

    public function getConfig() : string
    {
        if($this->blas===null)
            return 'PhpBlas';
        return $this->blas->getConfig();
    }

    public function getCorename() : string
    {
        if($this->blas===null)
            return 'PHP';
        return $this->blas->getCorename();
    }

    public function getParallel() : int
    {
        return 0; // parallel mode = 0 : serial 
    }

    public function hasIamin() : bool
    {
        return true;
    }

    public function hasOmatcopy() : bool
    {
        return true;
    }

    protected function sign(float $x,float $y) : float
    {
        if($y<0) {
            $x = -$x;
        }
        return $x;
    }

    /**
     * @param  int $trans BLAS::NoTrans, BLAS::Trans, BLAS::ConjTrans, BLAS::ConjNoTrans
     * @return array<bool> [bool $trans, bool $conj]
     */
    protected function codeToTrans(int $trans) : array
    {
        switch($trans) {
            case BLAS::NoTrans: {
                return [false,false];
            }
            case BLAS::Trans: {
                return [true,false];
            }
            case BLAS::ConjTrans: {
                return [true,true];
            }
            case BLAS::ConjNoTrans: {
                return [false,true];
            }
            default: {
                throw new InvalidArgumentException('Unknown Tranpose Code: '.$trans);
            }
        }
    }

    protected function cleanComplexNumber(mixed $value,string $name) : object
    {
        if(!$this->cisobject($value)) {
            throw new InvalidArgumentException("$name is not complex");
        }
        return $value;
    }

    protected function cleanFloatNumber(mixed $value,string $name) : float
    {
        if(!is_numeric($value)) {
            throw new InvalidArgumentException("$name is not float");
        }
        return (float)$value;
    }

    public function scal(
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idx = $offsetX;
        if($this->cistype($X->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            for ($i=0; $i<$n; $i++,$idx+=$incX) {
                $X[$idx] = $this->cmul($X[$idx],$alpha);
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            for ($i=0; $i<$n; $i++,$idx+=$incX) {
                $X[$idx] = $X[$idx] * $alpha;
            }
        }
    }
    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        if($this->cistype($X->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $Y[$idxY] = $this->cadd($this->cmul($alpha,$X[$idxX]),$Y[$idxY]);
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            if($alpha==1.0) {   // Y := X + Y
                for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                    $Y[$idxY] = $X[$idxX] + $Y[$idxY];
                }
            } else {            // Y := a*X + Y
                for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                    $Y[$idxY] = $alpha * $X[$idxX] + $Y[$idxY];
                }
            }
        }
    }

    public function dot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if($this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        if($this->cistype($X->dtype())) {
            $acc = $this->cbuild(0.0);
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $acc = $this->cadd($acc,$this->cmul($X[$idxX],$Y[$idxY]));
            }
        } else {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $acc += $X[$idxX] * $Y[$idxY];
            }
        }
        return $acc;
    }

    public function dotu(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if(!$this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = $this->cbuild(0.0);
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc = $this->cadd($acc,$this->cmul($X[$idxX],$Y[$idxY]));
        }
        return $acc;
    }

    public function dotc(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if(!$this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type:'.intval($X->dtype()));
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = $this->cbuild(0.0);
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc = $this->cadd($acc,$this->cmul($this->cconj($X[$idxX]),$Y[$idxY]));
        }
        return $acc;
    }

    public function asum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX;
        if($this->cistype($X->dtype())) {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $acc += $this->cabs($X[$idxX]);
            }
        } else {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $acc += abs($X[$idxX]);
            }
        }
        return $acc;
    }

    public function iamax(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX+$incX;
        $idx = 0;
        if($this->cistype($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = $this->cabs($X[$idxX]);
                if($acc < $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = abs($X[$idxX]);
                if($acc < $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    public function iamin(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX+$incX;
        $idx = 0;
        if($this->cistype($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = $this->cabs($X[$idxX]);
                if($acc > $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = abs($X[$idxX]);
                if($acc > $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    public function copy(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $Y[$idxY] = $X[$idxX];
        }
    }

    public function nrm2(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : float
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX;
        // Y := sqrt(sum(Xn ** 2))
        if($this->cistype($X->dtype())) {
            $sum = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $real = $X[$idxX]->real;
                $imag = $X[$idxX]->imag;
                $sum += $real*$real +  $imag*$imag;
            }
            $Y = sqrt($sum);
        } else {
            $sum = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $v = $X[$idxX];
                $sum += $v*$v;
            }
            $Y = sqrt($sum);
        }
        return $Y;
    }

    public function rotg(
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        $a = $A[$offsetA];
        $b = $B[$offsetB];
        $c = $s = $r = $z = 0.0;

        // Check if the data type of buffer A is complex.
        if(!$this->cistype($A->dtype())) {
            // Real number calculation block.
            $absa = abs($a);
            $absb = abs($b);
            // Case where the absolute value of b is zero.
            if($absb == 0.0) {
                $c = 1.0;
                $s = 0.0;
                $r = $a;
                $z =  0.0;
            // Case where the absolute value of a is zero.
            } elseif($absa == 0.0) {
                $c = 0.0;
                $s = 1.0;
                $r = $b;
                $z = 1.0;
            // General case where both absolute values are non-zero.
            } else {
                $safmin = 1.0e-37; // Smallest safe floating-point number.
                $safmax = 1/$safmin; // Largest safe floating-point number.
                // Scale factor to avoid overflow or underflow.
                $scale = min( max($safmin,max($absa,$absb)), $safmax);
                // Determine the sign for r.
                if ($absa > $absb) {
                    $sigma = ($a>=0.0)? 1.0:-1.0;
                } else {
                    $sigma = ($b>=0.0)? 1.0:-1.0;
                }

                $dascal = $a / $scale;
                $dbscal = $b / $scale;
                // Calculate r.
                $r_abs = ($scale * sqrt($dascal * $dascal + $dbscal * $dbscal));
                $r = $sigma * $r_abs;
                // Handle the case where r is zero (should be rare).
                if($r==0.0) { // Consider using a small tolerance instead of exact zero check
                    // If r is numerically zero, implies a=0 and b=0, handled above.
                    // This case might indicate issues if reached.
                    // Fallback to identity rotation might be safer depending on context.
                    $c = 1.0;
                    $s = 0.0;
                    $z = 0.0;
                    // $r remains 0.0 in this case
                } else {
                    // Calculate c and s.
                    $c = $a / $r;
                    $s = $b / $r;
                    $z = 1.0;
                    // Update z based on the relative magnitudes of abs(a) and abs(b).
                    if($absa > $absb) {
                        $z = $s;
                    } // Note: LAPACK d_rotg returns z=1/c if |a|<=|b| and c!=0
                    // Need to check if c=0 case needs explicit handling for z
                    elseif ($c != 0.0) { // Check if absa <= absb is needed here based on LAPACK doc/code
                       $z = 1.0 / $c;
                    } else {
                       // Case |a| <= |b| and c == 0 (implies a=0)
                       // Handled by initial elseif(absa == 0.0) case where z=1.0
                       // If somehow reached here (e.g., due to float precision), z=1.0 might be intended.
                       $z = 1.0; // Default or based on LAPACK behavior for this edge case
                    }
                }
            }

        } else {
            // Complex number calculation block.
            $calc = $this->getCalc($A->dtype());
            $zero_complex = $calc->build(0.0, 0.0);
            $one_complex = $calc->build(1.0, 0.0);
            $one_real = 1.0;
            $zero_real = 0.0;

            // Initial values (retrieve input a and b).
            $a_in = $a; // Keep the original a (if needed).
            $b_in = $b; // Keep the original b.

            // Case where a is zero and b is non-zero
            if ($calc->iszero($a_in)) {
                 if($calc->iszero($b_in)) {
                    // a = 0 and b = 0
                    $c_real = $one_real;
                    $s = $zero_complex;
                    $r = $zero_complex;
                 } else {
                    // a = 0 and b != 0
                    $c_real = $zero_real;
                    $s = $one_complex; // s = 1 + 0i
                    $r = $b_in;        // r = b
                 }
            } else {
                // Case where a is non-zero (b can be zero or non-zero).
                $absa = $calc->abs($a_in); // |a| (real number)
                $absb = $calc->abs($b_in); // |b| (real number)

                // Case 1: |a| > |b|
                if ($absa > $absb) {
                    $scale_real = $absa;
                    $roe_a = $calc->scale(1.0 / $scale_real, $a_in); // roe_a = a / |a|

                    $b_scaled = $calc->scale(1.0 / $scale_real, $b_in);
                    $abs_b_scaled = $calc->abs($b_scaled);
                    $norm_real = $scale_real * sqrt($one_real + $abs_b_scaled * $abs_b_scaled);

                    $c_real = $absa / $norm_real;
                    $b_div_norm = $calc->scale(1.0 / $norm_real, $b_in);
                    $s = $calc->mul($calc->conj($roe_a), $b_div_norm);
                    $r = $calc->scale($norm_real, $roe_a); // r = roe_a * norm

                // Case 2: |b| >= |a| (and a != 0).
                } else {
                    // Sub-case: |b| >= |a| > 0, but b is numerically zero.
                    // This is effectively the same as the b = 0 case (where a != 0).
                    if ($calc->iszero($b_in)) {
                        $c_real = $one_real;
                        $s = $zero_complex;
                        $r = $a_in; // r = a
                    } else {
                        // |b| >= |a| > 0 and b != 0.
                        $scale_real = $absb; // scaling also can be based b

                        // norm = |b| * sqrt(1 + (|a|/|b|)^2) = sqrt(|a|^2 + |b|^2).
                        $a_scaled = $calc->scale(1.0 / $scale_real, $a_in);
                        $abs_a_scaled = $calc->abs($a_scaled);
                        $norm_real = $scale_real * sqrt($one_real + $abs_a_scaled * $abs_a_scaled);

                        // c = |a| / norm (real number).
                        if ($norm_real == 0.0) {
                            // Should not happen if a!=0
                            $c_real = $one_real; // Safety fallback
                            $s = $zero_complex;
                            $r = $a_in;
                        } else {
                            $c_real = $absa / $norm_real;

                            // roe_a = a / |a| (unit vector with phase a)
                            // absa must be not zero. (because a!=0)
                            $roe_a = $calc->scale(1.0 / $absa, $a_in);

                            // r = roe_a * norm (Calculate r so that it has the topology of a)
                            $r = $calc->scale($norm_real, $roe_a);

                            // s = (a / |a|) * conj(b / norm) 
                            $b_div_norm = $calc->scale(1.0 / $norm_real, $b_in); // b / norm

                            $conj_b_div_norm = $calc->conj($b_div_norm);     // conj(b / norm)

                            // roe_a has already been calculated (a / |a|)
                            $s = $calc->mul($roe_a, $conj_b_div_norm);         // s = (a / |a|) * conj(b / norm)

                        }
                    }
                }
            }
            $c = $calc->build($c_real, 0.0); // c is real, but stored in a complex buffer.
        } // end of complex calculation block

        // Store results in buffers
        $A[$offsetA] = $r;
        if(!$this->cistype($A->dtype())) {
            $B[$offsetB] = $z; // Note: for complex, B is not used for 'z'
        }
        $C[$offsetC] = $c;
        $S[$offsetS] = $s;
    }

    public function rotmg(
        Buffer $D1, int $offsetD1,
        Buffer $D2, int $offsetD2,
        Buffer $B1, int $offsetB1,
        Buffer $B2, int $offsetB2,
        Buffer $P,  int $offsetP,
    ) : void
    {
        $dd1 = $D1[$offsetD1];
        $dd2 = $D2[$offsetD2];
        $dx1 = $B1[$offsetB1];
        $dy1 = $B2[$offsetB2];
        $dh11 = 0.0;
        $dh21 = 0.0;
        $dh12 = 0.0;
        $dh22 = 0.0;
        $dflag = -1;
        
        if($dd2 == 0.0 || $dy1 == 0.0)
        {
            $dflag = -2.0;
            $P[$offsetP+0] = $dflag;
            return;
        }
    
        if($dd1 < 0.0)
        {
            $dflag = -1.0;
            $dh11  = 0.0;
            $dh12  = 0.0;
            $dh21  = 0.0;
            $dh22  = 0.0;
    
            $dd1  = 0.0;
            $dd2  = 0.0;
            $dx1  = 0.0;
        }
        else if (($dd1 == 0.0 || $dx1 == 0.0) && $dd2 > 0.0)
        {
            $dflag = 1.0;
            $dh12 = 1;
            $dh21 = -1;
            $dx1 = $dy1;
            $dtemp = $dd1;
            $dd1 = $dd2;
            $dd2 = $dtemp;
        } 
        else
        {
            $dp2 = $dd2 * $dy1;
            $dp1 = $dd1 * $dx1;
            $dq2 =  $dp2 * $dy1;
            $dq1 =  $dp1 * $dx1;
            if(abs($dq1) > abs($dq2))
            {
                $dh11  =  1.0;
                $dh22  =  1.0;
                $dh21 = -  $dy1 / $dx1;
                $dh12 =    $dp2 /  $dp1;
    
                $du   = 1.0 - $dh12 * $dh21;
                $dflag = 0.0;
                $dd1  = $dd1 / $du;
                $dd2  = $dd2 / $du;
                $dx1  = $dx1 * $du;
                
            }
            else
            {
                if($dq2 < 0.0)
                {
                    $dflag = -1.0;
    
                    $dh11  = 0.0;
                    $dh12  = 0.0;
                    $dh21  = 0.0;
                    $dh22  = 0.0;
    
                    $dd1  = 0.0;
                    $dd2  = 0.0;
                    $dx1  = 0.0;
                }
                else
                {
                    $dflag =  1.0;
                    $dh21  = -1.0;
                    $dh12  =  1.0;
    
                    $dh11  =  $dp1 /  $dp2;
                    $dh22  = $dx1 /  $dy1;
                    $du    =  1.0 + $dh11 * $dh22;
                    $dtemp = $dd2 / $du;
    
                    $dd2  = $dd1 / $du;
                    $dd1  = $dtemp;
                    $dx1  = $dy1 * $du;
                }
            }
    
    
            while ( $dd1 <= self::RGAMSQ && $dd1 != 0.0)
            {
                $dflag = -1.0;
                $dd1  = $dd1 * (self::GAM * self::GAM);
                $dx1  = $dx1 / self::GAM;
                $dh11  = $dh11 / self::GAM;
                $dh12  = $dh12 / self::GAM;
            }
            while (abs($dd1) > self::GAMSQ) {
                $dflag = -1.0;
                $dd1  = $dd1 / (self::GAM * self::GAM);
                $dx1  = $dx1 * self::GAM;
                $dh11  = $dh11 * self::GAM;
                $dh12  = $dh12 * self::GAM;
            }
    
            while (abs($dd2) <= self::RGAMSQ && $dd2 != 0.0) {
                $dflag = -1.0;
                $dd2  = $dd2 * (self::GAM * self::GAM);
                $dh21  = $dh21 / self::GAM;
                $dh22  = $dh22 / self::GAM;
            }
            while (abs($dd2) > self::GAMSQ) {
                $dflag = -1.0;
                $dd2  = $dd2 / (self::GAM * self::GAM);
                $dh21  = $dh21 * self::GAM;
                $dh22  = $dh22 * self::GAM;
            }
    
        }
    
        if($dflag < 0.0)
        {
            $P[$offsetP+1] = $dh11;
            $P[$offsetP+2] = $dh21;
            $P[$offsetP+3] = $dh12;
            $P[$offsetP+4] = $dh22;
        }
        else
        {
            if($dflag == 0.0)
            {
                $P[$offsetP+2] = $dh21;
                $P[$offsetP+3] = $dh12;
            }
            else
            {
                $P[$offsetP+1] = $dh11;
                $P[$offsetP+4] = $dh22;
            }
        }
    
        $P[$offsetP+0] = $dflag;

        $D1[$offsetD1] = $dd1;
        $D2[$offsetD2] = $dd2;
        $B1[$offsetB1] = $dx1;
    }

    public function rot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if(!$this->cistype($X->dtype())) {
            $cc = $C[$offsetC];
            $ss = $S[$offsetS];
            $idX = $offsetX;
            $idY = $offsetY;
            for($i=0;$i<$n;$i++,$idX+=$incX,$idY+=$incY) {
                $xx = $X[$idX];
                $yy = $Y[$idY];
                $X[$idX] =  $cc * $xx + $ss * $yy;
                $Y[$idY] = -$ss * $xx + $cc * $yy;
            }
        } else {
            $calc = $this->getCalc($X->dtype());
            $cc_complex = $C[$offsetC];
            $ss = $S[$offsetS];
            $c_real = $cc_complex->real;

            $idX = $offsetX;
            $idY = $offsetY;

            //$c_complex_for_calc = $calc->build($c_real, 0.0);

            $zero_complex = $calc->build(0.0, 0.0);
            // $minus_one_complex = $calc->build(-1.0, 0.0);

            for($i=0; $i < $n; $i++, $idX += $incX, $idY += $incY) {
                $xx = $X[$idX];
                $yy = $Y[$idY];

                $temp_xx = $xx;

                // X[$idX] = c*xx + s*yy;
                $term1_x = $calc->scale($c_real, $xx);
                // s*yy
                $term2_x = $calc->mul($ss, $yy);
                $X[$idX] = $calc->add($term1_x, $term2_x);

                // Y[$idY] = -conj(s)*temp_xx + c*yy;
                // -conj(s)
                $conj_s = $calc->conj($ss);
                $neg_conj_s = $calc->sub($zero_complex, $conj_s); // 0 - conj(s)
                // or   $neg_conj_s = $calc->mul($minus_one_complex, $conj_s);

                // -conj(s)*temp_xx
                $term1_y = $calc->mul($neg_conj_s, $temp_xx);
                $term2_y = $calc->scale($c_real, $yy);
                $Y[$idY] = $calc->add($term1_y, $term2_y);
            }
        }
    }

    public function rotm(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $P,  int $offsetP,
    ) : void
    {
        $dflag = $P[$offsetP+0];
        if ($n <= 0 || $dflag == - 2.0) {
            // flag -2
            return;
        }
    
        $kx = $offsetX;
        $ky = $offsetY;
        if ($incX < 0) {
            $kx = (1 - $n) * $incX + $offsetX;
        }
        if ($incY < 0) {
            $ky = (1 - $n) * $incY + $offsetY;
        }
    
        if ($dflag < 0) {
            // flag = -1
            $dh11 = $P[$offsetP+1];
            $dh12 = $P[$offsetP+3];
            $dh21 = $P[$offsetP+2];
            $dh22 = $P[$offsetP+4];
            for ($i=0; $i<$n; ++$i) {
                $w = $X[$kx];
                $z = $Y[$ky];
                $X[$kx] = $w * $dh11 + $z * $dh12;
                $Y[$ky] = $w * $dh21 + $z * $dh22;
                $kx += $incX;
                $ky += $incY;
            }
        } elseif($dflag == 0) {
            // flag = 0
            $dh12 = $P[$offsetP+3];
            $dh21 = $P[$offsetP+2];
            for ($i=0; $i<$n; ++$i) {
                $w = $X[$kx];
                $z = $Y[$ky];
                $X[$kx] = $w + $z * $dh12;
                $Y[$ky] = $w * $dh21 + $z;
                $kx += $incX;
                $ky += $incY;
            }
        } else {
            // flag = 1
            $dh11 = $P[$offsetP+1];
            $dh22 = $P[$offsetP+4];
            for ($i=0; $i<$n; ++$i) {
                $w = $X[$kx];
                $z = $Y[$ky];
                $X[$kx] = $w * $dh11 + $z;
                $Y[$ky] = -$w + $dh22 * $z;
                $kx += $incX;
                $ky += $incY;
            }
        }
    }

    public function swap(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $tmp = $Y[$idxY];
            $Y[$idxY] = $X[$idxX];
            $X[$idxX] = $tmp;
        }
    }

    public function gemv(
        int $order,
        int $trans,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $rows = (!$trans) ? $m : $n;
        $cols = (!$trans) ? $n : $m;

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertMatrixBufferSpec("A", $A, $m, $n, $offsetA, $ldA);

        $this->assertVectorBufferSpec('X', $X, $cols, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $rows, $offsetY, $incY);

        $ldA_i = (!$trans) ? $ldA : 1;
        $ldA_j = (!$trans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idY = $offsetY;
        if($this->cistype($X->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
                $idA = $idA_i;
                $idX = $offsetX;
                $acc = $this->cbuild(0.0);
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                    // acc += alpha*A*X
                    $v = $A[$idA];
                    if($conj) {
                        $v = $this->cconj($v);
                    }
                    $v = $this->cmul($v,$X[$idX]);
                    if($hasAlpha) {
                        $v = $this->cmul($alpha,$v);
                    }
                    $acc = $this->cadd($acc,$v);
                }
                // Y = acc+beta*Y
                if($hasBeta) {
                    $v = $Y[$idY];
                    if($betaIsNotOne) {
                        $v = $this->cmul($beta,$v);
                    }
                    $acc = $this->cadd($acc,$v);
                }
                $Y[$idY] = $acc;
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            $hasBeta  = $beta!=0.0;
            for ($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
                $idA = $idA_i;
                $idX = $offsetX;
                $acc = 0.0;
                for ($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                    $acc += $alpha * $A[$idA] * $X[$idX];
                }
                if($hasBeta) {
                    $Y[$idY] = $acc + $beta * $Y[$idY];
                } else {
                    $Y[$idY] = $acc;
                }
            }
        }
    }

    private function accessArray(Buffer $A, int $offset, int $row, int $col, int $lda, int $order) : mixed
    {
        if($order == BLAS::RowMajor) {
            return $A[$offset + $row*$lda + $col];
        } else {
            return $A[$offset + $row + $col*$lda];
        }
    }

    private function accessIndex(int $offset, int $row, int $col, int $lda) : int
    {
        return $offset + $row*$lda + $col;
    }

    private function accessVector(Buffer $X, int $baseOffset, int $ix) : mixed
    {
        return $X[$baseOffset + $ix];
    }

    private function storeVector(Buffer $X, int $baseOffset, int $ix, mixed $value) : void
    {
        $X[$baseOffset + $ix] = $value;
    }

    private function getCalc(int $dtype) : object
    {
        if($this->cistype($dtype)) {
            return new PhpCalcComplex();
        } else {
            return new PhpCalcFloat();
        }
    }

    public function trsv(
        int $order,
        int $uplo,
        int $trans_code,
        int $diag,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
    ) : void
    {
        if ($n <= 0) {
            return; // 何もしない
        }

        [$trans, $conj] = $this->codeToTrans($trans_code);
        $dtype = $A->dtype();
        // echo "trans=";var_dump($trans);
        // echo "conj=";var_dump($conj);

        $notrans = !$trans;
        $nounit = ($diag == BLAS::NonUnit);
        $upper =  ($uplo == BLAS::Upper);
        $calc = $this->getCalc($dtype);
        $use_conj = $calc->iscomplex($dtype) && $conj;
        // echo "use_conj=";var_dump($use_conj);

        $kx = ($incX > 0) ? 0 : (1 - $n) * $incX;
        $forward_i_loop = ($notrans xor $upper);

        if ($forward_i_loop) {
            $start_i = 0; $end_i = $n; $inc_i = 1;
            $ix_start = $kx; $ix_inc = $incX;
        } else {
            $start_i = $n - 1; $end_i = -1; $inc_i = -1;
            $ix_start = $kx + ($n - 1) * $incX; $ix_inc = -$incX;
        }

        $ix = $ix_start;
        for ($i = $start_i; $i != $end_i; $i += $inc_i) {
            // A[i,i] を取得 (共役は取らない)
            $Aii = $this->accessArray($A, $offsetA, $i, $i, $ldA, $order);
            // 現在の X[i] (初期値は b[i]) を取得
            $temp = $this->accessVector($X, $offsetX, $ix);
            // echo "A[$i,$i]=Aii=$Aii,X[$ix]=temp=$temp\n";

            if ($notrans) { // NoTrans (A*x=b) または ConjNoTrans (conj(A)*x=b)
                //echo "notrans (or conjNoTrans)\n";
                $sum_val = $calc->build(0.0);
                if ($forward_i_loop) { // Lower (j < i)
                    //echo "forward_i_loop=true\n";
                    $jx = $kx;
                    for ($j = 0; $j < $i; $j++) {
                        $Aij = $this->accessArray($A, $offsetA, $i, $j, $ldA, $order); // A[i,j]
                        if ($use_conj) { // ConjNoTrans: conj(A[i,j]) を使う
                            $Aij = $calc->conj($Aij);
                        }
                        $xj = $this->accessVector($X, $offsetX, $jx); // 計算済みの x[j]
                        //echo "Use Aij(maybe conj)=$Aij, xj=$xj\n";
                        $sum_val = $calc->add($sum_val, $calc->mul($Aij, $xj));
                        $jx += $incX;
                    }
                } else { // Upper (j > i)
                    // echo "forward_i_loop=false\n";
                    $jx = $kx + ($i + 1) * $incX;
                    for ($j = $i + 1; $j < $n; $j++) {
                        $Aij = $this->accessArray($A, $offsetA, $i, $j, $ldA, $order); // A[i,j]
                        if ($use_conj) { // ConjNoTrans: conj(A[i,j]) を使う
                             $Aij = $calc->conj($Aij);
                        }
                        $xj = $this->accessVector($X, $offsetX, $jx); // 計算済みの x[j]
                        // echo "Use Aij(maybe conj)=$Aij, xj=$xj\n";
                        $sum_val = $calc->add($sum_val, $calc->mul($Aij, $xj));
                        $jx += $incX;
                    }
                }
                //echo "sum_val=$sum_val\n";
                $temp = $calc->sub($temp, $sum_val); // temp = b[i] - sum
                //echo "temp = b[i] - sum = $temp\n";

                if ($nounit) {
                    //echo "nounit\n";
                    $diag_val = $Aii; // オリジナルの A[i,i]
                    if ($use_conj) { // ConjNoTrans: conj(A[i,i]) で割る
                        $diag_val = $calc->conj($diag_val);
                    }
                    //echo "Use diag_val(maybe conj)=$diag_val\n";
                    if ($calc->iszero($diag_val)) {
                        // trigger_error("Matrix is singular at index $i", E_USER_WARNING);
                        $temp = NAN; // またはエラー処理
                    } else {
                         $temp = $calc->div($temp, $diag_val); // x[i] = temp / diag_val
                    }
                    //echo "x[i] = temp / diag_val = $temp\n";
                }
                //echo "SET X[$ix]=$temp\n";
                $this->storeVector($X, $offsetX, $ix, $temp); // x[i] を格納

            } else { // Trans (A^T*x=b) または ConjTrans (A^H*x=b)
                // echo "trans (or conjTrans)\n";
                $sum_val = $calc->build(0.0);
                if ($forward_i_loop) { // Upper (j < i)
                    // echo "forward_i_loop=true\n";
                    $jx = $kx;
                    for ($j = 0; $j < $i; $j++) {
                        $Aji = $this->accessArray($A, $offsetA, $j, $i, $ldA, $order); // A[j,i]
                        if ($use_conj) { // ConjTrans: conj(A[j,i]) を使う
                            $Aji = $calc->conj($Aji);
                        }
                        $xj = $this->accessVector($X, $offsetX, $jx); // 計算済みの x[j]
                        // echo "Use Aji(maybe conj)=$Aji, xj=$xj\n";
                        $sum_val = $calc->add($sum_val, $calc->mul($Aji, $xj));
                        $jx += $incX;
                    }
                } else { // Lower (j > i)
                    // echo "forward_i_loop=false\n";
                    $jx = $kx + ($i + 1) * $incX;
                    for ($j = $i + 1; $j < $n; $j++) {
                        $Aji = $this->accessArray($A, $offsetA, $j, $i, $ldA, $order); // A[j,i]
                        if ($use_conj) { // ConjTrans: conj(A[j,i]) を使う
                             $Aji = $calc->conj($Aji);
                        }
                        $xj = $this->accessVector($X, $offsetX, $jx); // 計算済みの x[j]
                        // echo "Use Aji(maybe conj)=$Aji, xj=$xj\n";
                        $sum_val = $calc->add($sum_val, $calc->mul($Aji, $xj));
                        $jx += $incX;
                    }
                }
                // echo "sum_val=$sum_val\n";
                $temp = $calc->sub($temp, $sum_val); // temp = b[i] - sum
                // echo "temp = b[i] - sum = $temp\n";

                if ($nounit) {
                    // echo "nounit\n";
                    $diag_val = $Aii; // オリジナルの A[i,i]
                    if ($use_conj) { // ConjTrans: conj(A[i,i]) で割る
                        $diag_val = $calc->conj($diag_val);
                    }
                    // echo "Use diag_val(maybe conj)=$diag_val\n";
                    if ($calc->iszero($diag_val)) {
                        // trigger_error("Matrix is singular at index $i", E_USER_WARNING);
                        $temp = NAN; // またはエラー処理
                    } else {
                        $temp = $calc->div($temp, $diag_val); // x[i] = temp / diag_val
                    }
                    // echo "x[i] = temp / diag_val = $temp\n";
                }
                // echo "SET X[$ix]=$temp\n";
                $this->storeVector($X, $offsetX, $ix, $temp); // x[i] を格納
            }

            $ix += $ix_inc; // 次の要素へ
        }
    }

    private function assertRowMajorOrder(int $order) : void
    {
        if($order != BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
    }

    private function assertBandParameter(string $name,int $k) : void
    {
        if($k<0) {
            throw new InvalidArgumentException("Argument $name must be greater than equals 0.");
        }
    }

    private function assertBandMatrixBufferSpec(
        string $name, Buffer $buffer,
        int $rows, int $ld, int $offset, int $bandRows) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<$bandRows) {
            throw new InvalidArgumentException("Argument ld$name must be greater than or equals to $bandRows.");
        }
        if($offset+$rows*$ld > count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }

    private function assertPackedMatrixBufferSpec(
        string $name, Buffer $buffer, int $n, int $offset) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        $size = intdiv($n*($n+1),2);
        if($offset+$size > count($buffer)) {
            throw new InvalidArgumentException("Packed matrix specification too large for buffer$name.");
        }
    }

    private function bandIndex(
        int $offset, int $ldA, int $i, int $j, int $kl) : int
    {
        return $offset + $i*$ldA + ($kl + $j - $i);
    }

    private function symmetricBandIndex(
        int $offset, int $ldA, int $i, int $j, int $k, bool $upper) : int
    {
        if($upper) {
            return $offset + $i*$ldA + ($j - $i);
        }
        return $offset + $i*$ldA + ($k + $j - $i);
    }

    private function packedIndex(int $i, int $j, int $n, bool $upper) : int
    {
        if($upper) {
            return $j + intdiv($i*(2*$n-$i-1),2);
        }
        return $j + intdiv($i*($i+1),2);
    }

    /**
     * @return array<mixed>
     */
    private function getAlphaBeta(
        int $dtype, object $calc,
        float|object $alpha, float|object $beta) : array
    {
        if($calc->iscomplex($dtype)) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
        }
        return [$alpha,$beta];
    }

    public function gbmv(
        int $order,
        int $trans_code,
        int $m,
        int $n,
        int $kl,
        int $ku,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        [$trans,$conj] = $this->codeToTrans($trans_code);
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertBandParameter('kl',$kl);
        $this->assertBandParameter('ku',$ku);

        $bandRows = $kl + $ku + 1;
        $this->assertBandMatrixBufferSpec('A',$A,$m,$ldA,$offsetA,$bandRows);
        $lenX = ($trans) ? $m : $n;
        $lenY = ($trans) ? $n : $m;
        $this->assertVectorBufferSpec('X',$X,$lenX,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$lenY,$offsetY,$incY);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);
        $useConj = $calc->iscomplex($dtype) && $conj;

        for($i=0; $i<$lenY; $i++) {
            $acc = $calc->build(0.0);
            $iy = $offsetY + $i*$incY;
            for($j=0; $j<$lenX; $j++) {
                if($trans) {
                    $row = $j; $col = $i;
                } else {
                    $row = $i; $col = $j;
                }
                if($col<$row-$kl || $col>$row+$ku) {
                    continue;
                }
                $idx = $this->bandIndex($offsetA,$ldA,$row,$col,$kl);
                $v = $A[$idx];
                if($useConj) {
                    $v = $calc->conj($v);
                }
                $acc = $calc->add($acc,$calc->mul($v,$X[$offsetX+$j*$incX]));
            }
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function sbmv(
        int $order,
        int $uplo,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertBandParameter('k',$k);
        $this->assertBandMatrixBufferSpec('A',$A,$n,$ldA,$offsetA,$k+1);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);

        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            $j0 = max(0,$i-$k);
            $j1 = min($n-1,$i+$k);
            for($j=$j0; $j<=$j1; $j++) {
                if($upper) {
                    $idx = ($j>=$i) ?
                        $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,true) :
                        $this->symmetricBandIndex($offsetA,$ldA,$j,$i,$k,true);
                } else {
                    $idx = ($j<=$i) ?
                        $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,false) :
                        $this->symmetricBandIndex($offsetA,$ldA,$j,$i,$k,false);
                }
                $acc = $calc->add($acc,$calc->mul($A[$idx],$X[$offsetX+$j*$incX]));
            }
            $iy = $offsetY + $i*$incY;
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function spmv(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $Ap, int $offsetAp,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);

        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            for($j=0; $j<$n; $j++) {
                if($upper) {
                    $idx = ($i<=$j) ?
                        $this->packedIndex($i,$j,$n,true) :
                        $this->packedIndex($j,$i,$n,true);
                } else {
                    $idx = ($i>=$j) ?
                        $this->packedIndex($i,$j,$n,false) :
                        $this->packedIndex($j,$i,$n,false);
                }
                $acc = $calc->add($acc,$calc->mul($Ap[$offsetAp+$idx],$X[$offsetX+$j*$incX]));
            }
            $iy = $offsetY + $i*$incY;
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function symv(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);

        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            for($j=0; $j<$n; $j++) {
                if($upper) {
                    $idx = ($i<=$j) ?
                        $this->accessIndex($offsetA,$i,$j,$ldA) :
                        $this->accessIndex($offsetA,$j,$i,$ldA);
                } else {
                    $idx = ($i>=$j) ?
                        $this->accessIndex($offsetA,$i,$j,$ldA) :
                        $this->accessIndex($offsetA,$j,$i,$ldA);
                }
                $acc = $calc->add($acc,$calc->mul($A[$idx],$X[$offsetX+$j*$incX]));
            }
            $iy = $offsetY + $i*$incY;
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function hemv(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);
        $isComplex = $calc->iscomplex($dtype);

        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            for($j=0; $j<$n; $j++) {
                if($i==$j) {
                    $v = $this->accessArray($A,$offsetA,$i,$j,$ldA,$order);
                    if($isComplex) {
                        $v = $calc->build($v->real);
                    }
                } elseif($upper) {
                    if($i<$j) {
                        $v = $this->accessArray($A,$offsetA,$i,$j,$ldA,$order);
                    } else {
                        $v = $this->accessArray($A,$offsetA,$j,$i,$ldA,$order);
                        $v = $calc->conj($v);
                    }
                } else {
                    if($i>$j) {
                        $v = $this->accessArray($A,$offsetA,$i,$j,$ldA,$order);
                    } else {
                        $v = $this->accessArray($A,$offsetA,$j,$i,$ldA,$order);
                        $v = $calc->conj($v);
                    }
                }
                $acc = $calc->add($acc,$calc->mul($v,$X[$offsetX+$j*$incX]));
            }
            $iy = $offsetY + $i*$incY;
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function hbmv(
        int $order,
        int $uplo,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertBandParameter('k',$k);
        $this->assertBandMatrixBufferSpec('A',$A,$n,$ldA,$offsetA,$k+1);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);
        $isComplex = $calc->iscomplex($dtype);

        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            $j0 = max(0,$i-$k);
            $j1 = min($n-1,$i+$k);
            for($j=$j0; $j<=$j1; $j++) {
                if($i==$j) {
                    $idx = $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,$upper);
                    $v = $A[$idx];
                    if($isComplex) {
                        $v = $calc->build($v->real);
                    }
                } elseif($upper) {
                    if($j>$i) {
                        $idx = $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,true);
                        $v = $A[$idx];
                    } else {
                        $idx = $this->symmetricBandIndex($offsetA,$ldA,$j,$i,$k,true);
                        $v = $calc->conj($A[$idx]);
                    }
                } else {
                    if($j<$i) {
                        $idx = $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,false);
                        $v = $A[$idx];
                    } else {
                        $idx = $this->symmetricBandIndex($offsetA,$ldA,$j,$i,$k,false);
                        $v = $calc->conj($A[$idx]);
                    }
                }
                $acc = $calc->add($acc,$calc->mul($v,$X[$offsetX+$j*$incX]));
            }
            $iy = $offsetY + $i*$incY;
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function hpmv(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $Ap, int $offsetAp,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha,$beta] = $this->getAlphaBeta($dtype,$calc,$alpha,$beta);
        $isComplex = $calc->iscomplex($dtype);

        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            for($j=0; $j<$n; $j++) {
                if($i==$j) {
                    $idx = $this->packedIndex($i,$j,$n,$upper);
                    $v = $Ap[$offsetAp+$idx];
                    if($isComplex) {
                        $v = $calc->build($v->real);
                    }
                } elseif($upper) {
                    if($i<$j) {
                        $idx = $this->packedIndex($i,$j,$n,true);
                        $v = $Ap[$offsetAp+$idx];
                    } else {
                        $idx = $this->packedIndex($j,$i,$n,true);
                        $v = $calc->conj($Ap[$offsetAp+$idx]);
                    }
                } else {
                    if($i>$j) {
                        $idx = $this->packedIndex($i,$j,$n,false);
                        $v = $Ap[$offsetAp+$idx];
                    } else {
                        $idx = $this->packedIndex($j,$i,$n,false);
                        $v = $calc->conj($Ap[$offsetAp+$idx]);
                    }
                }
                $acc = $calc->add($acc,$calc->mul($v,$X[$offsetX+$j*$incX]));
            }
            $iy = $offsetY + $i*$incY;
            $Y[$iy] = $calc->add($calc->mul($alpha,$acc),$calc->mul($beta,$Y[$iy]));
        }
    }

    public function trmv(
        int $order,
        int $uplo,
        int $trans_code,
        int $diag,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX ) : void
    {
        $this->assertRowMajorOrder($order);
        [$trans,$conj] = $this->codeToTrans($trans_code);
        $this->assertShapeParameter('n',$n);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);
        $unit = ($diag==BLAS::Unit);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        $useConj = $calc->iscomplex($dtype) && $conj;

        $src = [];
        for($j=0; $j<$n; $j++) {
            $src[$j] = $X[$offsetX+$j*$incX];
        }
        $dst = [];
        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            if(!$trans) {
                if($upper) { $j0 = $i; $j1 = $n-1; }
                else       { $j0 = 0; $j1 = $i; }
                for($j=$j0; $j<=$j1; $j++) {
                    if($j==$i && $unit) {
                        $v = $calc->build(1.0);
                    } else {
                        $v = $this->accessArray($A,$offsetA,$i,$j,$ldA,$order);
                        if($useConj) {
                            $v = $calc->conj($v);
                        }
                    }
                    $acc = $calc->add($acc,$calc->mul($v,$src[$j]));
                }
            } else {
                if($upper) { $j0 = 0; $j1 = $i; }
                else       { $j0 = $i; $j1 = $n-1; }
                for($j=$j0; $j<=$j1; $j++) {
                    if($j==$i && $unit) {
                        $v = $calc->build(1.0);
                    } else {
                        $v = $this->accessArray($A,$offsetA,$j,$i,$ldA,$order);
                        if($useConj) {
                            $v = $calc->conj($v);
                        }
                    }
                    $acc = $calc->add($acc,$calc->mul($v,$src[$j]));
                }
            }
            $dst[$i] = $acc;
        }
        for($i=0; $i<$n; $i++) {
            $X[$offsetX+$i*$incX] = $dst[$i];
        }
    }

    public function tbmv(
        int $order,
        int $uplo,
        int $trans_code,
        int $diag,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX ) : void
    {
        $this->assertRowMajorOrder($order);
        [$trans,$conj] = $this->codeToTrans($trans_code);
        $this->assertShapeParameter('n',$n);
        $this->assertBandParameter('k',$k);
        $this->assertBandMatrixBufferSpec('A',$A,$n,$ldA,$offsetA,$k+1);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);
        $unit = ($diag==BLAS::Unit);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        $useConj = $calc->iscomplex($dtype) && $conj;

        $src = [];
        for($j=0; $j<$n; $j++) {
            $src[$j] = $X[$offsetX+$j*$incX];
        }
        $dst = [];
        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            if(!$trans) {
                if($upper) { $j0 = $i; $j1 = min($n-1,$i+$k); }
                else       { $j0 = max(0,$i-$k); $j1 = $i; }
                for($j=$j0; $j<=$j1; $j++) {
                    if($j==$i && $unit) {
                        $v = $calc->build(1.0);
                    } else {
                        $idx = $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,$upper);
                        $v = $A[$idx];
                        if($useConj) {
                            $v = $calc->conj($v);
                        }
                    }
                    $acc = $calc->add($acc,$calc->mul($v,$src[$j]));
                }
            } else {
                if($upper) { $j0 = max(0,$i-$k); $j1 = $i; }
                else       { $j0 = $i; $j1 = min($n-1,$i+$k); }
                for($j=$j0; $j<=$j1; $j++) {
                    if($j==$i && $unit) {
                        $v = $calc->build(1.0);
                    } else {
                        $idx = $this->symmetricBandIndex($offsetA,$ldA,$j,$i,$k,$upper);
                        $v = $A[$idx];
                        if($useConj) {
                            $v = $calc->conj($v);
                        }
                    }
                    $acc = $calc->add($acc,$calc->mul($v,$src[$j]));
                }
            }
            $dst[$i] = $acc;
        }
        for($i=0; $i<$n; $i++) {
            $X[$offsetX+$i*$incX] = $dst[$i];
        }
    }

    public function tpmv(
        int $order,
        int $uplo,
        int $trans_code,
        int $diag,
        int $n,
        Buffer $Ap, int $offsetAp,
        Buffer $X, int $offsetX, int $incX ) : void
    {
        $this->assertRowMajorOrder($order);
        [$trans,$conj] = $this->codeToTrans($trans_code);
        $this->assertShapeParameter('n',$n);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);
        $unit = ($diag==BLAS::Unit);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        $useConj = $calc->iscomplex($dtype) && $conj;

        $src = [];
        for($j=0; $j<$n; $j++) {
            $src[$j] = $X[$offsetX+$j*$incX];
        }
        $dst = [];
        for($i=0; $i<$n; $i++) {
            $acc = $calc->build(0.0);
            if(!$trans) {
                if($upper) { $j0 = $i; $j1 = $n-1; }
                else       { $j0 = 0; $j1 = $i; }
                for($j=$j0; $j<=$j1; $j++) {
                    if($j==$i && $unit) {
                        $v = $calc->build(1.0);
                    } else {
                        $idx = $this->packedIndex($i,$j,$n,$upper);
                        $v = $Ap[$offsetAp+$idx];
                        if($useConj) {
                            $v = $calc->conj($v);
                        }
                    }
                    $acc = $calc->add($acc,$calc->mul($v,$src[$j]));
                }
            } else {
                if($upper) { $j0 = 0; $j1 = $i; }
                else       { $j0 = $i; $j1 = $n-1; }
                for($j=$j0; $j<=$j1; $j++) {
                    if($j==$i && $unit) {
                        $v = $calc->build(1.0);
                    } else {
                        $idx = $this->packedIndex($j,$i,$n,$upper);
                        $v = $Ap[$offsetAp+$idx];
                        if($useConj) {
                            $v = $calc->conj($v);
                        }
                    }
                    $acc = $calc->add($acc,$calc->mul($v,$src[$j]));
                }
            }
            $dst[$i] = $acc;
        }
        for($i=0; $i<$n; $i++) {
            $X[$offsetX+$i*$incX] = $dst[$i];
        }
    }

    public function tbsv(
        int $order,
        int $uplo,
        int $trans_code,
        int $diag,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX ) : void
    {
        $this->assertRowMajorOrder($order);
        [$trans,$conj] = $this->codeToTrans($trans_code);
        $this->assertShapeParameter('n',$n);
        $this->assertBandParameter('k',$k);
        $this->assertBandMatrixBufferSpec('A',$A,$n,$ldA,$offsetA,$k+1);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);
        $notrans = !$trans;
        $unit = ($diag==BLAS::Unit);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        $useConj = $calc->iscomplex($dtype) && $conj;

        $forward = ($notrans xor $upper);
        $start = ($forward) ? 0 : $n-1;
        $end   = ($forward) ? $n : -1;
        $step  = ($forward) ? 1 : -1;
        for($i=$start; $i!=$end; $i+=$step) {
            $sum = $calc->build(0.0);
            if($notrans) {
                if($upper) { $j0 = $i+1; $j1 = min($n-1,$i+$k); }
                else       { $j0 = max(0,$i-$k); $j1 = $i-1; }
                for($j=$j0; $j<=$j1; $j++) {
                    $idx = $this->symmetricBandIndex($offsetA,$ldA,$i,$j,$k,$upper);
                    $v = $A[$idx];
                    if($useConj) {
                        $v = $calc->conj($v);
                    }
                    $sum = $calc->add($sum,$calc->mul($v,$X[$offsetX+$j*$incX]));
                }
            } else {
                if($upper) { $j0 = max(0,$i-$k); $j1 = $i-1; }
                else       { $j0 = $i+1; $j1 = min($n-1,$i+$k); }
                for($j=$j0; $j<=$j1; $j++) {
                    $idx = $this->symmetricBandIndex($offsetA,$ldA,$j,$i,$k,$upper);
                    $v = $A[$idx];
                    if($useConj) {
                        $v = $calc->conj($v);
                    }
                    $sum = $calc->add($sum,$calc->mul($v,$X[$offsetX+$j*$incX]));
                }
            }
            $ix = $offsetX + $i*$incX;
            $temp = $calc->sub($X[$ix],$sum);
            if(!$unit) {
                $idx = $this->symmetricBandIndex($offsetA,$ldA,$i,$i,$k,$upper);
                $diagVal = $A[$idx];
                if($useConj) {
                    $diagVal = $calc->conj($diagVal);
                }
                if($calc->iszero($diagVal)) {
                    $temp = NAN;
                } else {
                    $temp = $calc->div($temp,$diagVal);
                }
            }
            $X[$ix] = $temp;
        }
    }

    public function tpsv(
        int $order,
        int $uplo,
        int $trans_code,
        int $diag,
        int $n,
        Buffer $Ap, int $offsetAp,
        Buffer $X, int $offsetX, int $incX ) : void
    {
        $this->assertRowMajorOrder($order);
        [$trans,$conj] = $this->codeToTrans($trans_code);
        $this->assertShapeParameter('n',$n);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);
        $notrans = !$trans;
        $unit = ($diag==BLAS::Unit);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        $useConj = $calc->iscomplex($dtype) && $conj;

        $forward = ($notrans xor $upper);
        $start = ($forward) ? 0 : $n-1;
        $end   = ($forward) ? $n : -1;
        $step  = ($forward) ? 1 : -1;
        for($i=$start; $i!=$end; $i+=$step) {
            $sum = $calc->build(0.0);
            if($notrans) {
                if($upper) { $j0 = $i+1; $j1 = $n-1; }
                else       { $j0 = 0; $j1 = $i-1; }
                for($j=$j0; $j<=$j1; $j++) {
                    $idx = $this->packedIndex($i,$j,$n,$upper);
                    $v = $Ap[$offsetAp+$idx];
                    if($useConj) {
                        $v = $calc->conj($v);
                    }
                    $sum = $calc->add($sum,$calc->mul($v,$X[$offsetX+$j*$incX]));
                }
            } else {
                if($upper) { $j0 = 0; $j1 = $i-1; }
                else       { $j0 = $i+1; $j1 = $n-1; }
                for($j=$j0; $j<=$j1; $j++) {
                    $idx = $this->packedIndex($j,$i,$n,$upper);
                    $v = $Ap[$offsetAp+$idx];
                    if($useConj) {
                        $v = $calc->conj($v);
                    }
                    $sum = $calc->add($sum,$calc->mul($v,$X[$offsetX+$j*$incX]));
                }
            }
            $ix = $offsetX + $i*$incX;
            $temp = $calc->sub($X[$ix],$sum);
            if(!$unit) {
                $idx = $this->packedIndex($i,$i,$n,$upper);
                $diagVal = $Ap[$offsetAp+$idx];
                if($useConj) {
                    $diagVal = $calc->conj($diagVal);
                }
                if($calc->iszero($diagVal)) {
                    $temp = NAN;
                } else {
                    $temp = $calc->div($temp,$diagVal);
                }
            }
            $X[$ix] = $temp;
        }
    }

    public function ger(
        int $order,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->gerx($order,$m,$n,$alpha,$X,$offsetX,$incX,$Y,$offsetY,$incY,
            $A,$offsetA,$ldA,false);
    }

    public function geru(
        int $order,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->gerx($order,$m,$n,$alpha,$X,$offsetX,$incX,$Y,$offsetY,$incY,
            $A,$offsetA,$ldA,false);
    }

    public function gerc(
        int $order,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->gerx($order,$m,$n,$alpha,$X,$offsetX,$incX,$Y,$offsetY,$incY,
            $A,$offsetA,$ldA,true);
    }

    private function gerx(
        int $order,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA,
        bool $conjY ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$m,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        $this->assertMatrixBufferSpec('A',$A,$m,$n,$offsetA,$ldA);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha] = $this->getAlphaBeta($dtype,$calc,$alpha,$calc->build(0.0));
        $useConj = $calc->iscomplex($dtype) && $conjY;

        for($i=0; $i<$m; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            for($j=0; $j<$n; $j++) {
                $yj = $Y[$offsetY+$j*$incY];
                if($useConj) {
                    $yj = $calc->conj($yj);
                }
                $idx = $this->accessIndex($offsetA,$i,$j,$ldA);
                $A[$idx] = $calc->add($A[$idx],$calc->mul($alpha,$calc->mul($xi,$yj)));
            }
        }
    }

    public function syr(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha] = $this->getAlphaBeta($dtype,$calc,$alpha,$calc->build(0.0));

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $idx = $this->accessIndex($offsetA,$i,$j,$ldA);
                $A[$idx] = $calc->add($A[$idx],
                    $calc->mul($alpha,$calc->mul($xi,$X[$offsetX+$j*$incX])));
            }
        }
    }

    public function spr(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Ap, int $offsetAp ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha] = $this->getAlphaBeta($dtype,$calc,$alpha,$calc->build(0.0));

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $idx = $this->packedIndex($i,$j,$n,$upper);
                $Ap[$offsetAp+$idx] = $calc->add($Ap[$offsetAp+$idx],
                    $calc->mul($alpha,$calc->mul($xi,$X[$offsetX+$j*$incX])));
            }
        }
    }

    public function syr2(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha] = $this->getAlphaBeta($dtype,$calc,$alpha,$calc->build(0.0));

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $yi = $Y[$offsetY+$i*$incY];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $xj = $X[$offsetX+$j*$incX];
                $yj = $Y[$offsetY+$j*$incY];
                $v = $calc->add($calc->mul($xi,$yj),$calc->mul($yi,$xj));
                $idx = $this->accessIndex($offsetA,$i,$j,$ldA);
                $A[$idx] = $calc->add($A[$idx],$calc->mul($alpha,$v));
            }
        }
    }

    public function spr2(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $Ap, int $offsetAp ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        [$alpha] = $this->getAlphaBeta($dtype,$calc,$alpha,$calc->build(0.0));

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $yi = $Y[$offsetY+$i*$incY];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $xj = $X[$offsetX+$j*$incX];
                $yj = $Y[$offsetY+$j*$incY];
                $v = $calc->add($calc->mul($xi,$yj),$calc->mul($yi,$xj));
                $idx = $this->packedIndex($i,$j,$n,$upper);
                $Ap[$offsetAp+$idx] = $calc->add($Ap[$offsetAp+$idx],$calc->mul($alpha,$v));
            }
        }
    }

    public function her(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        $isComplex = $calc->iscomplex($dtype);
        if($isComplex) {
            if(is_object($alpha)) {
                $alpha = $alpha->real;
            }
            $alpha = $calc->build((float)$alpha);
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
        }

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $xj = $X[$offsetX+$j*$incX];
                if($isComplex) {
                    $v = $calc->mul($xi,$calc->conj($xj));
                } else {
                    $v = $calc->mul($xi,$xj);
                }
                $idx = $this->accessIndex($offsetA,$i,$j,$ldA);
                if($i==$j && $isComplex) {
                    $real = $A[$idx]->real + $calc->mul($alpha,$v)->real;
                    $A[$idx] = $calc->build($real);
                } else {
                    $A[$idx] = $calc->add($A[$idx],$calc->mul($alpha,$v));
                }
            }
        }
    }

    public function hpr(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Ap, int $offsetAp ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        $isComplex = $calc->iscomplex($dtype);
        if($isComplex) {
            if(is_object($alpha)) {
                $alpha = $alpha->real;
            }
            $alpha = $calc->build((float)$alpha);
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
        }

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $xj = $X[$offsetX+$j*$incX];
                if($isComplex) {
                    $v = $calc->mul($xi,$calc->conj($xj));
                } else {
                    $v = $calc->mul($xi,$xj);
                }
                $idx = $this->packedIndex($i,$j,$n,$upper);
                if($i==$j && $isComplex) {
                    $real = $Ap[$offsetAp+$idx]->real + $calc->mul($alpha,$v)->real;
                    $Ap[$offsetAp+$idx] = $calc->build($real);
                } else {
                    $Ap[$offsetAp+$idx] = $calc->add($Ap[$offsetAp+$idx],$calc->mul($alpha,$v));
                }
            }
        }
    }

    public function her2(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        $this->assertMatrixBufferSpec('A',$A,$n,$n,$offsetA,$ldA);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $A->dtype();
        $calc = $this->getCalc($dtype);
        $isComplex = $calc->iscomplex($dtype);
        if($isComplex) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
        }
        $conjAlpha = ($isComplex) ? $calc->conj($alpha) : $alpha;

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $yi = $Y[$offsetY+$i*$incY];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $xj = $X[$offsetX+$j*$incX];
                $yj = $Y[$offsetY+$j*$incY];
                if($isComplex) {
                    $v = $calc->add(
                        $calc->mul($alpha,$calc->mul($xi,$calc->conj($yj))),
                        $calc->mul($conjAlpha,$calc->mul($yi,$calc->conj($xj))));
                } else {
                    $v = $calc->mul($alpha,
                        $calc->add($calc->mul($xi,$yj),$calc->mul($yi,$xj)));
                }
                $idx = $this->accessIndex($offsetA,$i,$j,$ldA);
                if($i==$j && $isComplex) {
                    $real = $A[$idx]->real + $v->real;
                    $A[$idx] = $calc->build($real);
                } else {
                    $A[$idx] = $calc->add($A[$idx],$v);
                }
            }
        }
    }

    public function hpr2(
        int $order,
        int $uplo,
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $Ap, int $offsetAp ) : void
    {
        $this->assertRowMajorOrder($order);
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X',$X,$n,$offsetX,$incX);
        $this->assertVectorBufferSpec('Y',$Y,$n,$offsetY,$incY);
        $this->assertPackedMatrixBufferSpec('Ap',$Ap,$n,$offsetAp);
        if($uplo!=BLAS::Upper && $uplo!=BLAS::Lower) {
            throw new InvalidArgumentException('Argument uplo must be Upper or Lower.');
        }
        $upper = ($uplo==BLAS::Upper);

        $dtype = $Ap->dtype();
        $calc = $this->getCalc($dtype);
        $isComplex = $calc->iscomplex($dtype);
        if($isComplex) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
        }
        $conjAlpha = ($isComplex) ? $calc->conj($alpha) : $alpha;

        for($i=0; $i<$n; $i++) {
            $xi = $X[$offsetX+$i*$incX];
            $yi = $Y[$offsetY+$i*$incY];
            $j0 = ($upper) ? $i : 0;
            $j1 = ($upper) ? $n-1 : $i;
            for($j=$j0; $j<=$j1; $j++) {
                $xj = $X[$offsetX+$j*$incX];
                $yj = $Y[$offsetY+$j*$incY];
                if($isComplex) {
                    $v = $calc->add(
                        $calc->mul($alpha,$calc->mul($xi,$calc->conj($yj))),
                        $calc->mul($conjAlpha,$calc->mul($yi,$calc->conj($xj))));
                } else {
                    $v = $calc->mul($alpha,
                        $calc->add($calc->mul($xi,$yj),$calc->mul($yi,$xj)));
                }
                $idx = $this->packedIndex($i,$j,$n,$upper);
                if($i==$j && $isComplex) {
                    $real = $Ap[$offsetAp+$idx]->real + $v->real;
                    $Ap[$offsetAp+$idx] = $calc->build($real);
                } else {
                    $Ap[$offsetAp+$idx] = $calc->add($Ap[$offsetAp+$idx],$v);
                }
            }
        }
    }

    /**
     * C(m,k) = A(m,n)*B(n,k)
     */
    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$transA,$conjA] = $this->codeToTrans($transA);
        [$transB,$conjB] = $this->codeToTrans($transB);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rowsA = (!$transA) ? $m : $k;
        $colsA = (!$transA) ? $k : $m;
        $rowsB = (!$transB) ? $k : $n;
        $colsB = (!$transB) ? $n : $k;

        $this->assertMatrixBufferSpec("A", $A, $rowsA, $colsA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $m, $n, $offsetC, $ldC);

        $ldA_m = (!$transA) ? $ldA : 1;
        $ldA_k = (!$transA) ? 1 : $ldA;
        $ldB_k = (!$transB) ? $ldB : 1;
        $ldB_n = (!$transB) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
                    $idA = $idA_m;
                    $idB = $idB_n;
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
                        $valueA = $A[$idA];
                        $valueB = $B[$idB];
                        if($conjA) {
                            $valueA = $this->cconj($valueA);
                        }
                        if($conjB) {
                            $valueB = $this->cconj($valueB);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$valueB));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
                    $idA = $idA_m;
                    $idB = $idB_n;
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
                        $acc += $A[$idA] * $B[$idB];
                    }
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function symm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        $sizeA = ($side==BLAS::Left) ? $m : $n;
        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $m, $n, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $m, $n, $offsetC, $ldC);

        $ldA_m = ($uplo==BLAS::Upper) ? $ldA : 1;
        $ldA_k = ($uplo==BLAS::Upper) ? 1 : $ldA;
        $ldB_k = ($side==BLAS::Left) ? $ldB : 1;
        $ldB_n = ($side==BLAS::Left) ? 1 : $ldB;
        $ldC_m = ($side==BLAS::Left) ? $ldC : 1;
        $ldC_n = ($side==BLAS::Left) ? 1 : $ldC;
        if($side==BLAS::Right) {
            [$n,$m] = [$m,$n];
        }

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for($im=0; $im<$m; $im++,$idC_m+=$ldC_m) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC+=$ldC_n) {
                    $idB = $idB_n;
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$sizeA; $ik++,$idB+=$ldB_k) {
                        if($ik<$im) {
                            $idA = $offsetA+$ik*$ldA_m+$im*$ldA_k;
                        } else {
                            $idA = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        }
                        $acc = $this->cadd($acc,$this->cmul($A[$idA],$B[$idB]));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$m; $im++,$idC_m+=$ldC_m) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC+=$ldC_n) {
                    $idB = $idB_n;
                    $acc = 0.0;
                    for ($ik=0; $ik<$sizeA; $ik++,$idB+=$ldB_k) {
                        if($ik<$im) {
                            $idA = $offsetA+$ik*$ldA_m+$im*$ldA_k;
                        } else {
                            $idA = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        }
                        $acc += $A[$idA] * $B[$idB];
                    }
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function syrk(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$n,$k] = [$k,$n];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rows = (!$trans) ? $n : $k;
        $cols = (!$trans) ? $k : $n;
        $this->assertMatrixBufferSpec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("C", $C, $n, $n, $offsetC, $ldC);

        $ldA_m  = (!$trans) ? $ldA : 1;
        $ldA_k  = (!$trans) ? 1 : $ldA;
        $ldAT_k = ($trans)  ? $ldA : 1;
        $ldAT_n = ($trans)  ? 1 : $ldA;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $valueA  = $A[$idA];
                        $valueAT = $A[$idAT];
                        if($conj) {
                            $valueA  = $this->cconj($valueA);
                            $valueAT = $this->cconj($valueAT);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$valueAT));
                    }
                    $idC = $im*$ldC+$in;
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $acc += $A[$idA] * $A[$idAT];
                    }
                    $idC = $im*$ldC+$in;
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function syr2k(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$n,$k] = [$k,$n];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rows = (!$trans) ? $n : $k;
        $cols = (!$trans) ? $k : $n;

        $this->assertMatrixBufferSpec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rows, $cols, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $n, $n, $offsetC, $ldC);

        $ldA_m  = (!$trans) ? $ldA : 1;
        $ldA_k  = (!$trans) ? 1 : $ldA;
        $ldAT_k = ($trans) ? $ldA : 1;
        $ldAT_n = ($trans) ? 1 : $ldA;
        $ldB_m  = (!$trans) ? $ldB : 1;
        $ldB_k  = (!$trans) ? 1 : $ldB;
        $ldBT_k = ($trans) ? $ldB : 1;
        $ldBT_n = ($trans) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+ $ik*$ldA_k;
                        $idB  = $offsetB+$im*$ldB_m+ $ik*$ldB_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $idBT = $offsetB+$ik*$ldBT_k+$in*$ldBT_n;
                        $valueA  = $A[$idA];
                        $valueAT = $A[$idAT];
                        if($conj) {
                            $valueA  = $this->cconj($valueA);
                            $valueAT = $this->cconj($valueAT);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$B[$idBT]));
                        $acc = $this->cadd($acc,$this->cmul($valueAT,$B[$idB]));
                    }
                    $idC = $im*$ldC+$in;
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+ $ik*$ldA_k;
                        $idB  = $offsetB+$im*$ldB_m+ $ik*$ldB_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $idBT = $offsetB+$ik*$ldBT_k+$in*$ldBT_n;
                        $acc += $A[$idA]  * $B[$idBT];
                        $acc += $A[$idAT] * $B[$idB];
                    }
                    $idC = $im*$ldC+$in;
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }


    /**
     *   B(m,n) = alpha * A(m,m)B(m,n)  : side=Left
     *   B(m,n) = alpha * B(m,n)A(n,n)  : side=Right
     */
    public function trmm(
        int $order,
        int $side,  // left or right
        int $uplo,  // upper or lower
        int $trans, // trans A
        int $diag,  // no unit or unit
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        if($side==BLAS::Left) {
            $sizeA = $m;
            $sizeB = $n;
            $right = false;
        } elseif($side==BLAS::Right) {
            $sizeA = $n;
            $sizeB = $m;
            $right = true;
        } else {
            throw new InvalidArgumentException('Invalid side value: '.$side);
        }
        if($uplo==BLAS::Upper) {
            $lower = false;
        } elseif($uplo==BLAS::Lower) {
            $lower = true;
        } else {
            throw new InvalidArgumentException('Invalid uplo value: '.$uplo);
        }
        if($diag==BLAS::NonUnit) {
            $unit = false;
        } elseif($diag==BLAS::Unit) {
            $unit = true;
        } else {
            throw new InvalidArgumentException('Invalid diag value: '.$diag);
        }
        $rowsB = $m;
        $colsB = $n;

        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);

        $trans = $right ? !$trans : $trans;
        $lower = $trans ? !$lower : $lower;
        $ldA_m = $trans ? 1 : $ldA;
        $ldA_k = $trans ? $ldA : 1;
        $ldB_k = $right ? 1 : $ldB;
        $ldB_n = $right ? $ldB : 1;

        $startm = $lower?($sizeA-1):0;
        $stepm =  $lower?(-1):1;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $hasAlpha = !$this->cisone($alpha);
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                for($in=0;$in<$sizeB;$in++) {
                    if($unit) {
                        $startk = $lower?0:$im+1;
                        $countk = $sizeA-$cm-1;
                        $acc = $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    } else {
                        $startk = $lower?0:$im;
                        $countk = $sizeA-$cm;
                        $acc = $this->cbuild(0.0);
                    }
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        $v = $A[$offsetA+$im*$ldA_m+$ik*$ldA_k];
                        if($conj) {
                            $v = $this->cconj($v);
                        }
                        $acc = $this->cadd($acc,$this->cmul($v,$B[$offsetB+$ik*$ldB_k+$in*$ldB_n]));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                for($in=0;$in<$sizeB;$in++) {
                    if($unit) {
                        $startk = $lower?0:$im+1;
                        $countk = $sizeA-$cm-1;
                        $acc = $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    } else {
                        $startk = $lower?0:$im;
                        $countk = $sizeA-$cm;
                        $acc = 0.0;
                    }
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        $acc += $A[$offsetA+$im*$ldA_m+$ik*$ldA_k]*$B[$offsetB+$ik*$ldB_k+$in*$ldB_n];
                    }
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $alpha * $acc;
                }
            }
        }
    }

    public function trsm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        if($side==BLAS::Left) {
            $sizeA = $m;
            $sizeB = $n;
            $right = false;
        } elseif($side==BLAS::Right) {
            $sizeA = $n;
            $sizeB = $m;
            $right = true;
        } else {
            throw new InvalidArgumentException('Invalid side value: '.$side);
        }
        if($uplo==BLAS::Upper) {
            $lower = false;
        } elseif($uplo==BLAS::Lower) {
            $lower = true;
        } else {
            throw new InvalidArgumentException('Invalid uplo value: '.$uplo);
        }
        if($diag==BLAS::NonUnit) {
            $unit = false;
        } elseif($diag==BLAS::Unit) {
            $unit = true;
        } else {
            throw new InvalidArgumentException('Invalid diag value: '.$diag);
        }
        $rowsB = $m;
        $colsB = $n;
        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);

        $trans = $right ? !$trans : $trans;
        $lower = $trans ? !$lower : $lower;
        $ldA_m = $trans ? 1 : $ldA;
        $ldA_k = $trans ? $ldA : 1;
        $ldB_k = $right ? 1 : $ldB;
        $ldB_n = $right ? $ldB : 1;

        $startm = $lower?0:($sizeA-1);
        $stepm =  $lower?1:(-1);
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $hasAlpha = !$this->cisone($alpha);
            // loop(i)
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                // A[i,i]
                if($unit) {
                    $denomi = 1.0;
                    $denomiFlag = 1; // is_one
                } else {
                    $denomi = $A[$offsetA+$im*$ldA_m+$im*$ldA_k];
                    if($this->ciszero($denomi)) {
                        $denomiFlag = 0; // is_zero
                        $denomi = $this->cbuild(NAN,i:NAN);
                    } else {
                        $denomiFlag = 2; // is_normal
                        if($conj) {
                            //echo "C";
                            $denomi = $this->cconj($denomi);
                        }
                    }
                }
                //echo "denomi:$denomi\n";
                // loop(j)
                //echo "for(j)[$in,$sizeB]\n";
                for($in=0;$in<$sizeB;$in++) {
                    // acc = 0;
                    $startk = $lower?0:($im+1);
                    $countk = $cm;
                    $acc = $this->cbuild(0.0);
                    // loop(k)
                    //echo "for(k)[$startk,$countk]\n";
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        // acc += A[i,k] * B[k,j];
                        //echo "a[$im,$ik]:";
                        //echo $A[$offsetA+$im*$ldA_m+$ik*$ldA_k].",";
                        //echo "b[$ik,$in],";
                        //echo "b[".($offsetB+$ik*$ldB_k+$in*$ldB_n)."]:";
                        //echo $B[$offsetB+$ik*$ldB_k+$in*$ldB_n].",";
                        $v = $A[$offsetA+$im*$ldA_m+$ik*$ldA_k];
                        if($conj) {
                            //echo "C";
                            $v = $this->cconj($v);
                        }
                        $acc = $this->cadd($acc,$this->cmul($v,$B[$offsetB+$ik*$ldB_k+$in*$ldB_n]));
                        //echo "acc:".$acc.",";
                    }
                    //echo "endfor(k)\n";
                    //echo "acc:".$acc.",";
                    //echo "\n";
                    // B[i,j] = (B[i,j] - acc) / A[i,i];
                    //echo "B[$im,$in]";
                    //echo "[$ldB_k,$ldB_n]";
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($denomiFlag==0) { // NAN
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $denomi;
                    } elseif($denomiFlag==1) { // denomi == 1.0
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $this->csub($B[$offsetB+$im*$ldB_k+$in*$ldB_n],$acc);
                    } else {
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $this->cdiv($this->csub($B[$offsetB+$im*$ldB_k+$in*$ldB_n],$acc),$denomi);
                    }
                    //echo "B[".($offsetB+$im*$ldB_k+$in*$ldB_n)."]:";
                    //echo $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    //echo "\n";
                }
                //echo "endfor(j)\n";
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            // loop(i)
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                // A[i,i]
                if($unit) {
                    $denomi = 1.0;
                } else {
                    $denomi = $A[$offsetA+$im*$ldA_m+$im*$ldA_k];
                    if($denomi==0) {
                        $denomi = NAN;
                    }
                }
                // loop(j)
                for($in=0;$in<$sizeB;$in++) {
                    // acc = 0;
                    $startk = $lower?0:($im+1);
                    $countk = $cm;
                    $acc = 0.0;
                    // loop(k)
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        // acc += A[i,k] * B[k,j];
                        $acc += $A[$offsetA+$im*$ldA_m+$ik*$ldA_k]*$B[$offsetB+$ik*$ldB_k+$in*$ldB_n];
                    }
                    // B[i,j] = (B[i,j] - acc) / A[i,i];
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = ($B[$offsetB+$im*$ldB_k+$in*$ldB_n] - $alpha*$acc) / $denomi;
                }
            }
        }
    }

    public function omatcopy(
        int $order,
        int $trans,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
    ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $rows = (!$trans) ? $m : $n;
        $cols = (!$trans) ? $n : $m;

        $this->assertMatrixBufferSpec("A", $A, $m, $n, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rows, $cols, $offsetB, $ldB);

        // Check Buffer A and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }

        $ldA_i = (!$trans) ? $ldA : 1;
        $ldA_j = (!$trans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idB_i = $offsetB;

        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $hasAlpha = !$this->cisone($alpha);
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idB_i+=$ldB) {
                $idA = $idA_i;
                $idB = $idB_i;
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idB++) {
                    $v = $A[$idA];
                    if($conj) {
                        $v = $this->cconj($v);
                    }
                    if($hasAlpha) {
                        $v = $this->cmul($alpha,$v);
                    }
                    $B[$idB] = $v;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idB_i+=$ldB) {
                $idA = $idA_i;
                $idB = $idB_i;
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idB++) {
                    $B[$idB] = $alpha * $A[$idA];
                }
            }
        }
    }
    
}
