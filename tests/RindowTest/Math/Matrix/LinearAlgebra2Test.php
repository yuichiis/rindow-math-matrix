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


    public function testSolve()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [1, 1, 1],
            [2, 4, 6],
            [2, 0, 4],
        ]);
        $b = $la->array(
             [10, 38, 14]
        );
        $solve = $la->solve($a,$b);
        //echo $mo->toString($solve,'%f',true);
        $this->assertEquals([3,5,2],$solve->toArray());
    }

    public function testIsInt()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $this->assertTrue($la->isInt($la->array(1,dtype:NDArray::int8)));
        $this->assertTrue($la->isInt($la->array(1,dtype:NDArray::uint8)));
        $this->assertTrue($la->isInt($la->array(1,dtype:NDArray::int32)));
        $this->assertTrue($la->isInt($la->array(1,dtype:NDArray::uint32)));
        $this->assertTrue($la->isInt($la->array(1,dtype:NDArray::int64)));
        $this->assertTrue($la->isInt($la->array(1,dtype:NDArray::uint64)));

        $this->assertFalse($la->isInt($la->array(1,dtype:NDArray::float32)));
        $this->assertFalse($la->isInt($la->array(1,dtype:NDArray::float64)));

        $this->assertFalse($la->isInt($la->array(1,dtype:NDArray::bool)));
    }

}
