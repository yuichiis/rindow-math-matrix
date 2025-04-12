<?php
namespace RindowTest\Math\Matrix\LinearAlgebra2Test;

use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\Attributes\DataProvider;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\DeviceBuffer;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Matrix\Drivers\Selector;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Plot\Plot;
use ArrayObject;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;
use function Rindow\Math\Matrix\C;

use Rindow\Math\Matrix\Drivers\MatlibExt;

class LinearAlgebra2Test extends TestCase
{
    static protected $speedtest = false;
    protected $equalEpsilon = 1e-04;
    protected $service;

    protected $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
        NDArray::complex64=>'complex64', NDArray::complex128=>'complex128',
    ];

    public function setUp() : void
    {
        $selector = new Selector();
        $this->service = $selector->select();
    }

    public function newMatrixOperator()
    {
        $mo = new MatrixOperator(service:$this->service);
        //if($service->serviceLevel()<Service::LV_ADVANCED) {
        //    throw new \Exception("the service is not Advanced.");
        //}
        return $mo;
    }

    public function newLA($mo)
    {
        return $mo->la();
    }

    public function newArray(array $shape, ?int $dtype=null)
    {
        if($dtype===null)
            $dtype = NDArray::float32;
        $array = new NDArrayPhp(null,$dtype,$shape,service:$this->service);
        $size = $array->size();
        $buffer = $array->buffer();
        for($i=0;$i<$size;$i++) {
            $buffer[$i] = 0.0;
        }
        return $array;
    }

    public function ndarray($x)
    {
        return $x;
    }

    public function equalTest($a,$b)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($a instanceof NDArray) {
            if(!($b instanceof NDArray))
                throw new InvalidArgumentException('NDArrays must be of the same type.');
            if($a->shape()!=$b->shape())
                return false;
            $delta = $la->zerosLike($b);
            $la->copy($b,$delta);
            $la->axpy($a,$delta,-1.0);
            $delta = $la->asum($delta);
        } elseif(is_numeric($a)) {
            if(!is_numeric($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = abs($a - $b);
        } elseif(is_bool($a)) {
            if(!is_bool($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = ($a==$b)? 0 : 1;
        } else {
            throw new InvalidArgumentException('Values must be DNArray or float or int.');
        }

        if($delta < $this->equalEpsilon) {
            return true;
        } else {
            return false;
        }
    }

    protected function toComplex(mixed $array) : mixed
    {
        if(!is_array($array)) {
            if(is_numeric($array)) {
                return C($array,i:0);
            } else {
                return C($array->real,i:$array->imag);
            }
        }
        $cArray = [];
        foreach($array as $value) {
            $cArray[] = $this->toComplex($value);
        }
        return $cArray;
    }

    public function testSpeedTest()
    {
        // ==============================================
        // The speed test should normally be False.
        // Temporarily change to True only when performing
        // the corresponding test individually.
        // ==============================================
        $this->assertFalse(self::$speedtest);
    }


    public function testSvdSmallU()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
        [$u,$s,$vt] = $la->svd($a,$full_matrices=false);

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $la->array([
            [-0.59, 0.26, 0.36, 0.31, 0.23],
            [-0.40, 0.24,-0.22,-0.75,-0.36],
            [-0.03,-0.60,-0.45, 0.23,-0.31],
            [-0.43, 0.24,-0.69, 0.33, 0.16],
            [-0.47,-0.35, 0.39, 0.16,-0.52],
            [ 0.29, 0.58,-0.02, 0.38,-0.65],
        ]);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
    }

}
