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

    public function testEqualNormalLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::bool));
        $Y = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::bool));
        $Z = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::bool));
        $this->assertEquals($Z->toArray(),$la->equal($X,$Y)->toArray());

        $X = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::int32));
        $Y = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::int32));
        $Z = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::int32));
        $this->assertEquals($Z->toArray(),$la->equal($X,$Y)->toArray());

        $X = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::float32));
        $Y = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::float32));
        $Z = $la->fill(1,$la->alloc([100,1000],dtype:NDArray::float32));
        $this->assertEquals($Z->toArray(),$la->equal($X,$Y)->toArray());
    }

    public function testNotEqualNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $la->array([100,10,-1000]);
        $Y = $la->array([100,-10,-1000]);
        $this->assertEquals([0,1,0],$la->notEqual($X,$Y)->toArray());

        $X = $la->array([100,10,-1000],dtype:NDArray::int32);
        $Y = $la->array([100,-10,-1000],dtype:NDArray::int32);
        $this->assertEquals([0,1,0],$la->notEqual($X,$Y)->toArray());

        $X = $la->array([true,true,false],dtype:NDArray::bool);
        $Y = $la->array([true,false,false],dtype:NDArray::bool);
        $this->assertEquals([false,true,false],$la->notEqual($X,$Y)->toArray());
    }

    public function testNotNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $la->array([100,0,-1000]);
        $this->assertEquals([0,1,0],$la->not($X)->toArray());

        $X = $la->array([100,0,-1000],dtype:NDArray::int32);
        $this->assertEquals([0,1,0],$la->not($X)->toArray());

        $X = $la->array([true,true,false],dtype:NDArray::bool);
        $this->assertEquals([false,false,true],$la->not($X)->toArray());
    }

    public function testSoftmax()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $y = $la->softmax($x);
        if(is_scalar($y[0][0])) {
            $this->assertTrue($this->equalTest(0.05801,$y[0][0]));
            $this->assertTrue($this->equalTest(0.09564,$y[0][1]));
            $this->assertTrue($this->equalTest(0.15769,$y[0][2]));
            $this->assertTrue($this->equalTest(0.25999,$y[0][3]));
            $this->assertTrue($this->equalTest(0.42865,$y[0][4]));
        } else {
            $this->assertTrue($this->equalTest(0.05801,$y[0][0]->toArray()));
            $this->assertTrue($this->equalTest(0.09564,$y[0][1]->toArray()));
            $this->assertTrue($this->equalTest(0.15769,$y[0][2]->toArray()));
            $this->assertTrue($this->equalTest(0.25999,$y[0][3]->toArray()));
            $this->assertTrue($this->equalTest(0.42865,$y[0][4]->toArray()));
        }
    }

    public function testSoftmaxLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
            // large size
        $colsize = 100000;//600000;
        $rowsize = 64;
        $x = $la->ones($la->alloc([$rowsize,$colsize],dtype:NDArray::float32));
        $trues = $la->fill(1/$colsize,$la->alloc([$rowsize,$colsize],dtype:NDArray::float32));
        $r = $la->softmax($x);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));

        // large size
        $colsize = 64;
        $rowsize = 100000;//800000;
        $x = $la->ones($la->alloc([$rowsize,$colsize],dtype:NDArray::float32));
        $trues = $la->fill(1/$colsize,$la->alloc([$rowsize,$colsize],dtype:NDArray::float32));
        $r = $la->softmax($x);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));
    }

    public function testSoftmaxSpeed()
    {
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }

        // large size
        $colsize = 600000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],dtype:NDArray::float32);
        fwrite(STDERR,"fill-start\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end\n");
        fwrite(STDERR,"pre-start\n");
        $r = $la->softmax($x);
        fwrite(STDERR,"pre-end\n");
        $start = hrtime(true);
        $r = $la->softmax($x);
        $end = hrtime(true);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        // large size
        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],dtype:NDArray::float32);
        fwrite(STDERR,"fill-start\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end\n");
        fwrite(STDERR,"pre-start\n");
        $r = $la->softmax($x);
        fwrite(STDERR,"pre-end\n");
        $start = hrtime(true);
        $r = $la->softmax($x);
        $end = hrtime(true);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }

    public function testTopKNormal10()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D
        $size = 10;
        $input = $mo->arange($size,dtype:NDArray::float32);
        $input = $la->array($input);
        $k = 3;

        [$values, $indices] = $la->topK($input, k:$k);

        $this->assertEquals([ 9,  8,  7],$values->toArray());
        $this->assertEquals($input->dtype(),$values->dtype());
        $this->assertEquals([ 9,  8,  7],$indices->toArray());
        $this->assertEquals(NDArray::int32,$indices->dtype());

        // k=1
        $size = 10;
        $input = $mo->arange($size,dtype:NDArray::float32);
        $input = $la->array($input);

        [$values, $indices] = $la->topK($input);

        $this->assertEquals([ 9 ],$values->toArray());
        $this->assertEquals($input->dtype(),$values->dtype());
        $this->assertEquals([ 9 ],$indices->toArray());
        $this->assertEquals(NDArray::int32,$indices->dtype());

        // 2D
        $batches = 3;
        $size = 10;
        $input = $mo->arange($batches*$size,dtype:NDArray::float32)
            ->reshape([$batches,$size]);
        $input = $la->array($input);
        $k = 3;

        //$input = $la->array([[1, 0, 9, 8, 4, 3, 2,  7, 6, 5]]);
        //$k = 10;

        [$values, $indices] = $la->topK($input, k:$k);

        $this->assertEquals([
            [ 9,  8,  7],
            [19, 18, 17],
            [29, 28, 27]
        ],$values->toArray());
        $this->assertEquals($input->dtype(),$values->dtype());

        $this->assertEquals([
            [ 9,  8,  7],
            [ 9,  8,  7],
            [ 9,  8,  7],
        ],$indices->toArray());
        $this->assertEquals(NDArray::int32,$indices->dtype());

        // 3D
        $batches = 3;
        $heads = 2;
        $size = 10;
        $input = $mo->arange($batches*$heads*$size,dtype:NDArray::float32)
            ->reshape([$batches,$heads,$size]);
        $input = $la->array($input);
        $k = 3;

        [$values, $indices] = $la->topK($input, k:$k);

        $this->assertEquals([
            [[ 9,  8,  7],
             [19, 18, 17]],
            [[29, 28, 27],
             [39, 38, 37]],
            [[49, 48, 47],
             [59, 58, 57]],
        ],$values->toArray());
        $this->assertEquals($input->dtype(),$values->dtype());

        $this->assertEquals([
            [[ 9,  8,  7],
             [ 9,  8,  7]],
            [[ 9,  8,  7],
             [ 9,  8,  7]],
            [[ 9,  8,  7],
             [ 9,  8,  7]],
        ],$indices->toArray());
        $this->assertEquals(NDArray::int32,$indices->dtype());

    }

    public function testTopKNormal5000()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 3;
        $size = 5000;
        $k = 10;
        $input = $la->randomUniform(
            [$batches,$size],
            0, 1000,
            dtype:NDArray::float32,
        );
        $inputArray = $input->toArray();
        [$values,$indices] = $la->topK($input,k:$k);
        $values = $la->toNDArray($values);
        $indices = $la->toNDArray($indices);
        $SortedInput = [];
        foreach($inputArray as $inp) {
            arsort($inp,SORT_NUMERIC);
            $SortedInput[] = $inp;
        }
        $i = 0;
        foreach($SortedInput as $sortedInp) {
            $j = 0;
            foreach ($sortedInp as $topIndex => $topInp) {
                if($topInp!=$values[$i][$j]) {
                    $this->assertEquals($topInp,$values[$i][$j]);
                    break;
                }
                if($topInp!=$inputArray[$i][$indices[$i][$j]]) {
                    $this->assertEquals($topIndex,$indices[$i][$j]);
                    break;
                }
                $j++;
                if($j>=$k) {
                    break;
                }
            }
            $i++;
        }
        $this->assertTrue(true);
    }

    public function testTopkWithoutSorted()
    {
        $dtype = NDArray::float32;
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 2;
        $size = 5000;
        $k = 10;
        $sorted = false;

        $input = $la->randomUniform(
            [$batches,$size],
            0, 1000,
            dtype:NDArray::float32,
        );
        $inputArray = $input->toArray();
        [$values,$indices] = $la->topK($input,k:$k,sorted:$sorted);
        $values = $la->toNDArray($values);
        $indices = $la->toNDArray($indices);
        $SortedInput = [];
        foreach($inputArray as $inp) {
            arsort($inp,SORT_NUMERIC);
            $SortedInput[] = $inp;
        }

        $SortedInput = [];
        foreach($inputArray as $inp) {
            arsort($inp,SORT_NUMERIC);
            $SortedInput[] = $inp;
        }

        $unmatch = false;
        $i = 0;
        foreach($SortedInput as $sortedInp) {
            $j = 0;
            foreach ($sortedInp as $topIndex => $topInp) {
                if($topInp!=$values[$i][$j]) {
                    $unmatch = true;
                    break;
                }
                if($topInp!=$inputArray[$i][$indices[$i][$j]]) {
                    $unmatch = true;
                    break;
                }
                $j++;
                if($j>=$k) {
                    break;
                }
            }
            $i++;
        }
        $this->assertTrue($unmatch);

        $valuesArray = $values->toArray();
        $indicesArray = $indices->toArray();
        $i = 0;
        foreach($SortedInput as $sortedInp) {
            $j = 0;
            foreach ($sortedInp as $topIndex => $topInp) {
                if(!in_array($topInp,$valuesArray[$i])) {
                    var_dump($valuesArray[$i]);
                    $this->assertEquals($topInp,'notfound');
                }
                if(!in_array($topIndex,$indicesArray[$i])) {
                    var_dump($indicesArray[$i]);
                    $this->assertEquals($topIndex,'notfound');
                }
                $j++;
                if($j>=$k) {
                    break;
                }
            }
            $i++;
        }
    }

    public function testastype()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $math = $la;

        #### int to any
        $X = $la->array([-1,0,1,2,3],dtype:NDArray::int32);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals(NDArray::float32,$Y->dtype());
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        if($la->fp64()) {
            $dtype = NDArray::float64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([-1,0,1,2,3],$Y->toArray());
        }

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([255,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());
        $this->assertEquals(NDArray::bool,$Y->dtype());
        $this->assertEquals(4,$math->scalar($math->sum($Y)));
        $Z = $math->astype($Y, NDArray::int32);  // check reverse
        $this->assertEquals([1,0,1,1,1],$Z->toArray());
        $this->assertEquals(NDArray::int32,$Z->dtype());

        #### float to any ######
        $X = $la->array([-1,0,1,2,3],dtype:NDArray::float32);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        if($la->fp64()) {
            $dtype = NDArray::float64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([-1,0,1,2,3],$Y->toArray());
        }

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());
        $this->assertEquals(NDArray::bool,$Y->dtype());
        $Z = $math->astype($Y, NDArray::float32);  // check reverse
        $this->assertEquals([1,0,1,1,1],$Z->toArray());
        $this->assertEquals(NDArray::float32,$Z->dtype());

        #### bool to any ######
        $X = $la->array([true,false,true,true,true],dtype:NDArray::bool);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        if($la->fp64()) {
            $dtype = NDArray::float64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([1,0,1,1,1],$Y->toArray());
        }

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        if($la->accelerated()) {
            $devType = $math->getContext()->getInfo(OpenCL::CL_CONTEXT_DEVICES)->getInfo(0,OpenCL::CL_DEVICE_TYPE);
            $clVersion = $la->getCLVersion();
            $devName = $math->getContext()->getInfo(OpenCL::CL_CONTEXT_DEVICES)->getInfo(0,OpenCL::CL_DEVICE_NAME);
        } else {
            $devType = OpenCL::CL_DEVICE_TYPE_CPU;
            $clVersion = null;
            $devName = "CPU";
        }
        #### float to unsigned ######
        $X = $la->array([-1,0,1,2,3],dtype:NDArray::float32);
        $dtype = NDArray::uint8;
        $Y = $math->astype($X, $dtype);
        if($devType===OpenCL::CL_DEVICE_TYPE_GPU) {
            if($devName=='Loveland') {
                $this->assertEquals([255,0,1,2,3],$Y->toArray());// AMD GPU on Windows
            } else {
                $this->assertEquals([0,0,1,2,3],$Y->toArray());
            }
        } else {
            $this->assertEquals([255,0,1,2,3],$Y->toArray());
        }

        $dtype = NDArray::uint16;
        $Y = $math->astype($X, $dtype);
        if($devType===OpenCL::CL_DEVICE_TYPE_GPU) {
            if($devName=='Loveland') {
                $this->assertEquals([65535,0,1,2,3],$Y->toArray());
            } else {
                $this->assertEquals([0,0,1,2,3],$Y->toArray());
            }
        } else {
            $this->assertEquals([65535,0,1,2,3],$Y->toArray());
        }

        // ***** CAUTION ******
        $X = $la->array([-1000,0,1,2,4294967295],dtype:NDArray::float32);
        if($la->accelerated()) {
            // GPU
            $dtype = NDArray::uint32;
            $Y = $math->astype($X, $dtype);
            if($devType===OpenCL::CL_DEVICE_TYPE_GPU) {
                $this->assertEquals([0,0,1,2,4294967295],$Y->toArray());
            } else {
                $this->assertEquals([4294966296,0,1,2,0],$Y->toArray());
            }
        } else {
            // CPU
            $dtype = NDArray::uint32;
            $Y = $math->astype($X, $dtype);
            if($mo->service()->serviceLevel()>=Service::LV_ADVANCED) {
                $this->assertEquals([4294966296,0,1,2,0],$Y->toArray());
            } else {
                $this->assertEquals([4294966296,0,1,2,4294967295],$Y->toArray());
            }
        }

        // ***** CAUTION ******
        $X = $la->array([-1000,0,1,2,3],dtype:NDArray::float32);
        if($la->accelerated()) {
            // GPU
            $dtype = NDArray::uint64;;
            $Y = $math->astype($X, $dtype);
            $Yvalue = $Y->toArray();
            if($devType===OpenCL::CL_DEVICE_TYPE_GPU) {
                if($Yvalue==[0,0,1,2,3]) {
                    $this->assertTrue(true);
                } elseif($Yvalue==[1000,0,1,2,3]) {
                    $this->assertTrue(true);
                } elseif($Yvalue==[-1000,0,1,2,3]) {
                    $this->assertTrue(true);
                } else {
                    $this->assertTrue(false);
                }
            } else {
                $this->assertEquals([-1000,0,1,2,3],$Y->toArray());
            }
        } elseif($la->getConfig()=='PhpBlas') {
            // CPU
            $dtype = NDArray::uint64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([-1000,0,1,2,3],$Y->toArray());
        }
    }

    public static function providerIm2col2dNormal()
    {
        return [
            'normal' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h channels_first' => [[  // fail on mesa
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_h cols_channels_first' => [[ // fail on mesa
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_w cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_h cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_w cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_h cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_w cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => true,
            ]],
        ];
    }

    public static function providerIm2col2dNormalDEBUG()
    {
        return [
            'kernel_h' => [[
                'batches' => 1,//2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 1,//3,
                'kernel_h' => 4,//4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
        ];
    }

    #[DataProvider('providerIm2col2dNormal')]
    public function testIm2col2dNormal($params)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        extract($params);

        //$batches = 1;
        //$im_h = 4;
        //$im_w = 4;
        //$channels = 3;
        //$kernel_h = 3;
        //$kernel_w = 3;
        //$stride_h = 1;
        //$stride_w = 1;
        //$padding = null;
        //$channels_first = null;
        //$dilation_h = 1;
        //$dilation_w = 1;
        //$cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_h,
                $im_w
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_h,
                $im_w,
                $channels
            ]);
        }
        $cols = $la->im2col(
            $images,
            filterSize:[
                $kernel_h,$kernel_w],
            strides:[
                $stride_h,$stride_w],
            padding:$padding,
            channels_first:$channels_first,
            dilation_rate:[
                $dilation_h,$dilation_w],
            cols_channels_first:$cols_channels_first
        );
        $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $padding_h = (int)intdiv((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1),2);
            $padding_w = (int)intdiv((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1),2);
            $out_h = $im_h;
            $out_w = $im_w;
        } else {
            $padding_h = 0;
            $padding_w = 0;
        }

        if($cols_channels_first) {
            $this->assertEquals(
                [
                    $batches,
                    $out_h,$out_w,
                    $channels,
                    $kernel_h,$kernel_w,
                ],
                $cols->shape()
            );
        } else {
            $this->assertEquals(
                [
                    $batches,
                    $out_h,$out_w,
                    $kernel_h,$kernel_w,
                    $channels,
                ],
                $cols->shape()
            );
        }
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_y=0;$im_y<$out_h;$im_y++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                            for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                if($channels_first) {
                                    $input_id = ((($batch_id*$channels+$channel_id)*$im_h+$input_y)*$im_w+$input_x);
                                } else {
                                    $input_id = ((($batch_id*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                }
                                if($cols_channels_first) {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$channels+$channel_id)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                } else {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                }
                                if($input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                    $truesBuffer[$cols_id] = $input_id;
                                }
                            }
                        }
                    }
                }
            }
        }
        #echo "======================================\n";
        #echo $mo->toString($trues,'%3.0f',false)."\n";

        #echo "======================================\n";
        #echo $mo->toString($cols,'%3.0f',false)."\n";
        #return;

        $this->assertEquals($trues->toArray(),$cols->toArray());
        // channels_first kernel stride
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
        //        echo "kernel_h=$kernel_y\n";
        //        for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
        //            echo "kernel_w=$kernel_x\n";
        //            for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //                echo "channel=$channel_id\n";
        //                for($out_y=0;$out_y<$out_h;$out_y++) {
        //                    echo "[";
        //                    for($out_x=0;$out_x<$out_w;$out_x++) {
        //                        $value = $cols[$batch_id][$out_y][$out_x][$kernel_y][$kernel_x][$channel_id];
        //                        if(!is_scalar($value)) { $value = $value->toArray(); }
        //                        echo sprintf('%2d',intval($value)).",";
        //                    }
        //                    echo "],\n";
        //                }
        //                echo "\n";
        //            }
        //        }
        //    }
        //}

        // channels_first kernel last
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //        for($out_y=0;$out_y<$out_h;$out_y++) {
        //            echo "col_h=$out_y\n";
        //            for($out_x=0;$out_x<$out_w;$out_x++) {
        //                echo "col_w=$out_x\n";
        //                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
        //                    echo "[";
        //                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
        //                        $value = $cols[$batch_id][$out_y][$out_x][$kernel_y][$kernel_x][$channel_id];
        //                        if(!is_scalar($value))
        //                            $value = $value->toArray();
        //                        echo sprintf('%3d',$value).",";
        //                    }
        //                    echo "],";
        //                    echo "\n";
        //                }
        //            }
        //        }
        //    }
        //}

        //foreach ($cols->toArray() as $batch) {
        //    foreach ($batch as $col_h_id => $col_h_value) {
        //        echo "stride_h=$col_h_id\n";
        //        foreach ($col_h_value as $col_w_id => $col_w_value) {
        //            echo "stride($col_h_id,$col_w_id)\n";
        //            foreach ($col_w_value as $key => $kernel_h_value) {
        //                #echo "kernel_h=$key\n";
        //                foreach ($kernel_h_value as $key => $kernel_w_value) {
        //                    #echo "kernel_w=$key\n";
        //                    echo "[";
        //                    foreach ($kernel_w_value as $key => $channel_value) {
        //                            echo sprintf('%2d',$channel_value).",";
        //                    }
        //                    echo "],";
        //                }
        //                echo "\n";
        //            }
        //        }
        //    }
        //}
        //echo "==== cols trues ======\n";
        //foreach ($trues->toArray() as $batch) {
        //    foreach ($batch as $col_h_id => $col_h_value) {
        //        echo "stride_h=$col_h_id\n";
        //        foreach ($col_h_value as $col_w_id => $col_w_value) {
        //            echo "stride($col_h_id,$col_w_id)\n";
        //            foreach ($col_w_value as $key => $kernel_h_value) {
        //                #echo "kernel_h=$key\n";
        //                foreach ($kernel_h_value as $key => $kernel_w_value) {
        //                    #echo "kernel_w=$key\n";
        //                    echo "[";
        //                    foreach ($kernel_w_value as $key => $channel_value) {
        //                            echo sprintf('%2d',$channel_value).",";
        //                    }
        //                    echo "],";
        //                }
        //                echo "\n";
        //            }
        //        }
        //    }
        //}
        //$this->assertTrue(false);
        //$this->assertEquals(
        //[[
        //  [
        //   [[[ 0, 1, 2],[ 3, 4, 5],[ 6, 7, 8]],
        //    [[12,13,14],[15,16,17],[18,19,20]],
        //    [[24,25,26],[27,28,29],[30,31,32]],],
        //   [[[ 3, 4, 5],[ 6, 7, 8],[ 9,10,11]],
        //    [[15,16,17],[18,19,20],[21,22,23]],
        //    [[27,28,29],[30,31,32],[33,34,35]],],
        //  ],
        //  [
        //   [[[12,13,14],[15,16,17],[18,19,20]],
        //    [[24,25,26],[27,28,29],[30,31,32]],
        //    [[36,37,38],[39,40,41],[42,43,44]],],
        //   [[[15,16,17],[18,19,20],[21,22,23]],
        //    [[27,28,29],[30,31,32],[33,34,35]],
        //    [[39,40,41],[42,43,44],[45,46,47]],],
        //  ],
        //]],
        //$cols->toArray()
        //);

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            filterSize:[
                $kernel_h,$kernel_w],
            strides:[
                $stride_h,$stride_w],
            padding:$padding,
            channels_first:$channels_first,
            dilation_rate:[
                $dilation_h,$dilation_w],
            cols_channels_first:$cols_channels_first
        );
        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
        //foreach ($newImages->toArray() as $batch) {
        //    foreach ($batch as $key => $im_y) {
        //        #echo "kernel_h=$key\n";
        //        foreach ($im_y as $key => $im_x) {
        //            #echo "kernel_w=$key\n";
        //            echo "[";
        //            foreach ($im_x as $key => $channel) {
        //                    echo sprintf('%3d',$channel).",";
        //            }
        //            echo "],";
        //        }
        //        echo "\n";
        //    }
        //}
        // channels_first
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //        echo "channel=$channel_id\n";
        //        for($im_y=0;$im_y<$im_h;$im_y++) {
        //            echo "[";
        //            for($im_x=0;$im_x<$im_w;$im_x++) {
        //                echo sprintf('%3d',intval($newImages[$batch_id][$im_y][$im_x][$channel_id]->toArray())).",";
        //            }
        //            echo "],\n";
        //        }
        //        echo "\n";
        //    }
        //}

        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_y=0;$im_y<$out_h;$im_y++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                            for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                if($channels_first) {
                                    $input_id = ((($batch_id*$channels+$channel_id)*$im_h+$input_y)*$im_w+$input_x);
                                } else {
                                    $input_id = ((($batch_id*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                }
                                if($cols_channels_first) {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$channels+$channel_id)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                } else {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                }
                                if($input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                    $value = $imageBuffer[$input_id];
                                    $imageBuffer[$input_id] = $value + $truesBuffer[$cols_id];
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($imagesTrues->toArray(),$newImages->toArray());
        //echo "==== reverse trues ======\n";
        //foreach ($imagesTrues->toArray() as $batch) {
        //    foreach ($batch as $key => $im_y) {
        //        #echo "kernel_h=$key\n";
        //        foreach ($im_y as $key => $im_x) {
        //            #echo "kernel_w=$key\n";
        //            echo "[";
        //            foreach ($im_x as $key => $channel) {
        //                    echo sprintf('%3d',$channel).",";
        //            }
        //            echo "],";
        //        }
        //        echo "\n";
        //    }
        //}
    }

    public function testIm2col2dLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }

        $batches = 1;
        $im_h = 1024;
        $im_w = 1024;
        $channels = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ));
        //$images = $la->array($mo->ones(
        //    [$batches*
        //    $im_h*$im_w*
        //    $channels],
        //    NDArray::float32
        //));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_h,
                $im_w
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_h,
                $im_w,
                $channels
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        //var_dump($cols->shape());
        #echo "===== cols =====\n";
        #echo $mo->toString($cols->toNDArray(),'%2d',true);
        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $this->assertTrue(true);
    }

    public function testIm2col2dForPool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_h = 2;
        $kernel_w = 2;
        $stride_h = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=true;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ))->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $out_h = $im_h;
            $out_w = $im_w;
        }

        $this->assertEquals(
            [
                $batches,
                $out_h,$out_w,
                $channels,
                $kernel_h,$kernel_w,
            ],
            $cols->shape()
        );
        $this->assertEquals(
        [[
          [
           [[[0,3],[12,15]],
            [[1,4],[13,16]],
            [[2,5],[14,17]],],
           [[[6,9],[18,21]],
            [[7,10],[19,22]],
            [[8,11],[20,23]],],
          ],
          [
           [[[24,27],[36,39]],
            [[25,28],[37,40]],
            [[26,29],[38,41]],],
           [[[30,33],[42,45]],
            [[31,34],[43,46]],
            [[32,35],[44,47]],],
          ],
        ]],
        $cols->toArray()
        );

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
    }

    public static function providerIm2col1dNormal()
    {
        return [
            'normal' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'kernel_w' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'stride_w' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'dilation_w' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 2,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'normal padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => true,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'kernel_w padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => true,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'stride_w padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => true,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'dilation_w padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => true,
                'dilation_w' => 2,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'normal channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'kernel_w channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'stride_w channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'dilation_w channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 2,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'normal cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
            'kernel_w cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
            'stride_w cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
            'dilation_w cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 2,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
        ];
    }

    #[DataProvider('providerIm2col1dNormal')]
    public function testIm2col1dNormal($params)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        extract($params);

        //$batches = 1;
        //$im_w = 4;
        //$channels = 3;
        //$kernel_w = 3;
        //$stride_w = 1;
        //$padding = null;
        //$dilation_w = 1;
        //$channels_first = null;
        //$cols_channels_first = null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_w,
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_w,
                $channels,
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );
        $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $padding_w = intdiv((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1),2);
            $out_w = $im_w;
        } else {
            $padding_w = 0;
        }

        if($cols_channels_first) {
            $this->assertEquals(
                [
                    $batches,
                    $out_w,
                    $channels,
                    $kernel_w,
                ],
                $cols->shape()
            );
        } else {
            $this->assertEquals(
                [
                    $batches,
                    $out_w,
                    $kernel_w,
                    $channels,
                ],
                $cols->shape()
            );
        }
        //$this->assertEquals(
        //[[
        //   [[0,1,2],[3,4,5],[6,7,8]],
        //   [[3,4,5],[6,7,8],[9,10,11]],
        //]],
        //$cols->toArray()
        //);
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                        if($channels_first) {
                            $input_id = (($batch_id*$channels+$channel_id)*$im_w+$input_x);
                        } else {
                            $input_id = (($batch_id*$im_w+$input_x)*$channels+$channel_id);
                        }
                        if($cols_channels_first) {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$channels+$channel_id)*$kernel_w+$kernel_x);
                        } else {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$kernel_w+$kernel_x)*$channels+$channel_id);
                        }
                        if($input_x>=0 && $input_x<$im_w) {
                            $truesBuffer[$cols_id] = $input_id;
                        }
                    }
                }
            }
        }
        $this->assertEquals($trues->toArray(),$cols->toArray());

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);

        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                        if($channels_first) {
                            $input_id = (($batch_id*$channels+$channel_id)*$im_w+$input_x);
                        } else {
                            $input_id = (($batch_id*$im_w+$input_x)*$channels+$channel_id);
                        }
                        if($cols_channels_first) {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$channels+$channel_id)*$kernel_w+$kernel_x);
                        } else {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$kernel_w+$kernel_x)*$channels+$channel_id);
                        }
                        if($input_x>=0 && $input_x<$im_w) {
                            $value = $imageBuffer[$input_id];
                            $imageBuffer[$input_id] = $value + $truesBuffer[$cols_id];
                        }
                    }
                }
            }
        }
        $this->assertEquals($imagesTrues->toArray(),$newImages->toArray());
    }

    public function testIm2col1dForPool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 1;
        $im_w = 4;
        $channels = 3;
        $kernel_w = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $dilation_w = 1;
        $cols_channels_first=true;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ))->reshape([
            $batches,
            $im_w,
            $channels
        ]);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );
        $out_w = 2;

        $this->assertEquals(
            [
                $batches,
                $out_w,
                $channels,
                $kernel_w,
            ],
            $cols->shape()
        );
        $this->assertEquals(
          [
           [[[0,3],[1,4],[2,5]],
            [[6,9],[7,10],[8,11]]],
          ],
        $cols->toArray()
        );

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
    }
    public static function providerIm2col3dNormal()
    {
        return [
            'normal' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_d' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_d' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_d' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_d channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h channels_first' => [[ // fail on mesa
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_d channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_d channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],

            'stride_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_d cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_h cols_channels_first' => [[ // fail on mesa
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_w cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_d cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_h cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_w cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_d cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_h cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_w cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => true,
            ]],
        ];
    }


    public static function providerIm2col3dNormalDEBUG()
    {
        return [
            'kernel_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
        ];
    }

    #[DataProvider('providerIm2col3dNormal')]
    public function testIm2col3dNormal($params)
    {
        //return;
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        extract($params);

        //$batches = 1;
        //$im_d = 5;
        //$im_h = 5;
        //$im_w = 5;
        //$channels = 3;
        //$kernel_d = 3;
        //$kernel_h = 3;
        //$kernel_w = 3;
        //$stride_d = 1;
        //$stride_h = 1;
        //$stride_w = 1;
        //$padding = null;
        //$channels_first = null;
        //$dilation_d = 2;
        //$dilation_h = 1;
        //$dilation_w = 1;
        //$cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_d,
                $im_h,
                $im_w,
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_d,
                $im_h,
                $im_w,
                $channels,
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_d = intdiv(($im_d-($kernel_d-1)*$dilation_d-1),$stride_d)+1;
        $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $padding_d = intdiv((($im_d-1)*$stride_d-$im_d+($kernel_d-1)*$dilation_d+1),2);
            $padding_h = intdiv((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1),2);
            $padding_w = intdiv((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1),2);
            $out_d = $im_d;
            $out_h = $im_h;
            $out_w = $im_w;
        } else {
            $padding_d = 0;
            $padding_h = 0;
            $padding_w = 0;
        }

        if($cols_channels_first) {
            $this->assertEquals(
                [
                    $batches,
                    $out_d,$out_h,$out_w,
                    $channels,
                    $kernel_d,$kernel_h,$kernel_w,
                ],
                $cols->shape()
            );
        } else {
            $this->assertEquals(
                [
                    $batches,
                    $out_d,$out_h,$out_w,
                    $kernel_d,$kernel_h,$kernel_w,
                    $channels,
                ],
                $cols->shape()
            );
        }
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_z=0;$im_z<$out_d;$im_z++) {
                    for($im_y=0;$im_y<$out_h;$im_y++) {
                        for($im_x=0;$im_x<$out_w;$im_x++) {
                            for($kernel_z=0;$kernel_z<$kernel_d;$kernel_z++) {
                                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                        $input_z = $im_z*$stride_d+$kernel_z*$dilation_d-$padding_d;
                                        $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                        if($channels_first) {
                                            $input_id = (((($batch_id*$channels+$channel_id)*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x);
                                        } else {
                                            $input_id = (((($batch_id*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                        }
                                        if($cols_channels_first) {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$channels+$channel_id)*$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                        } else {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                        }
                                        if($input_z>=0 && $input_z<$im_d && $input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                            $truesBuffer[$cols_id] = $input_id;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($trues->toArray(),$cols->toArray());
        //$this->assertEquals(
        //    [],$cols->toArray()
        //);
        //$this->assertNotEquals(
        //    $la->zerosLike($cols)->toArray(),
        //    $cols->toArray()
        //);
        //foreach ($cols->toArray() as $batch) {
        //    foreach ($batch as $im_z => $im_z_value) {
        //        echo "im_z=$im_z\n";
        //        foreach ($im_z_value as $im_y => $im_y_value) {
        //            echo "im_y=$im_y\n";
        //            foreach ($im_y_value as $im_x => $im_x_value) {
        //                echo "im($im_z,$im_y,$im_x)\n";
        //                foreach ($im_x_value as $kernel_z => $kernel_z_value) {
        //                    foreach ($kernel_z_value as $kernel_y => $kernel_y_value) {
        //                        foreach ($kernel_y_value as $kernel_x => $kernel_x_value) {
        //                            echo "[";
        //                            foreach ($kernel_x_value as $channel_id => $channel_value) {
        //                                    echo sprintf('%3d',$channel_value).",";
        //                            }
        //                            echo "],";
        //                        }
        //                        echo "\n";
        //                    }
        //                    echo "\n";
        //                }
        //            }
        //        }
        //    }
        //}

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_z=0;$im_z<$out_d;$im_z++) {
                    for($im_y=0;$im_y<$out_h;$im_y++) {
                        for($im_x=0;$im_x<$out_w;$im_x++) {
                            for($kernel_z=0;$kernel_z<$kernel_d;$kernel_z++) {
                                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                        $input_z = $im_z*$stride_d+$kernel_z*$dilation_d-$padding_d;
                                        $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                        if($channels_first) {
                                            $input_id = (((($batch_id*$channels+$channel_id)*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x);
                                        } else {
                                            $input_id = (((($batch_id*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                        }
                                        if($cols_channels_first) {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$channels+$channel_id)*$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                        } else {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                        }
                                        if($input_z>=0 && $input_z<$im_d && $input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                            $value = $imageBuffer[$input_id];
                                            $imageBuffer[$input_id] = $value + $truesBuffer[$cols_id];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($imagesTrues->toArray(),$newImages->toArray());
    }

    public function testIm2col3dForPool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 1;
        $im_d = 4;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_d = 2;
        $kernel_h = 2;
        $kernel_w = 2;
        $stride_d = 2;
        $stride_h = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $dilation_d = 1;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=true;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            dtype:NDArray::float32
        ))->reshape([
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_d = intdiv(($im_d-($kernel_d-1)*$dilation_d-1),$stride_d)+1;
        $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;

        $this->assertEquals(
            [
                $batches,
                $out_d,$out_h,$out_w,
                $channels,
                $kernel_d,$kernel_h,$kernel_w,
            ],
            $cols->shape()
        );
        //$this->assertEquals(
        //[],
        //$cols->toArray()
        //);
        $this->assertNotEquals(
            $la->zerosLike($cols)->toArray(),
            $cols->toArray()
        );

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
    }

    public function testIm2col2dSpeed()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }
        echo "\n";
        $batches = 8;
        $im_h = 512;
        $im_w = 512;
        $channels = 3;
        $images = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->ones($images);
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=null;
        $cols = null;
        echo "im=($im_h,$im_w),knl=($kernel_h,$kernel_w),batches=$batches\n";

        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $newImages = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $batches = 256;
        $im_h = 28;
        $im_w = 28;
        $channels = 1;
        $images = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->ones($images);
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=null;
        $cols = null;
        echo "im=($im_h,$im_w),knl=($kernel_h,$kernel_w),batches=$batches\n";

        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $newImages = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        $this->assertTrue(true);
    }

    public function testRandomUniform()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->randomUniform(
            $shape=[20,30],
            $low=-1.0,
            $high=1.0);
        $y = $la->randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1);
        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $this->assertLessThanOrEqual(1,$la->max($x));
        $this->assertGreaterThanOrEqual(-1,$la->min($x));
        $fluct = $this->chi2($x->size(), $la->toNDArray($x), $la->min($x), $la->max($x), 10);
        $this->assertLessThan(50.0,$fluct); // 16.919>fluct

        $x = $la->randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            $dtype=NDArray::int32
            );
        $y = $la->randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            $dtype=NDArray::int32);
        $this->assertEquals(
            NDArray::int32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $x = $la->astype($x,NDArray::float32);
        if($this->service->serviceLevel()>=Service::LV_ADVANCED) {
            $this->assertEquals(1,$la->max($x));
            $this->assertEquals(-1,$la->min($x));
        } else {
            $this->assertEquals(1,round($la->max($x)));
            $this->assertEquals(-1,round($la->min($x)));
        }
        $fluct = $this->chi2($x->size(), $la->toNDArray($x), $la->min($x), $la->max($x), 3);
        $this->assertLessThan(50.0,$fluct); // 16.919>fluct
    }

    protected function chi2(int $n, NDArray $X, float $min, float $max, int $density) : float
    {
        $xx = $X->reshape([(int)array_product($X->shape())]);
        $hist_size = $density;
        $histogram = [];
        for($i=0;$i<$hist_size;++$i) {
            $histogram[$i] = 0;
        }

        for($i=0;$i<$n;++$i) {
            $index = (int)(($xx[$i]-$min)/(($max-$min)/$hist_size));
            if($index>=$hist_size) {
                $index = $hist_size-1;
            }
            $histogram[$index]++;
        }
        $chi_square = 0.0;
        for($i = 0; $i < $hist_size; $i++) {
            $chi_square += ($histogram[$i] - $n/$hist_size) * ($histogram[$i] - $n/$hist_size) / ($n/$hist_size);
        }
        return $chi_square;
    }

    public function testRandomNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->randomNormal(
            $shape=[20,30],
            $mean=0.0,
            $scale=1.0);
        $y = $la->randomNormal(
            $shape=[20,30],
            $mean=0.0,
            $scale=1.0);
        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $this->assertLessThanOrEqual(6,$la->max($x));
        $this->assertGreaterThanOrEqual(-6,$la->min($x));
    }

    public function testRandomSequence()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->randomSequence(
            $base=500,
            $size=100
            );
        $y = $la->randomSequence(
            $base=500,
            $size=100
            );
        if($la->accelerated()) {
            $this->assertEquals(
                NDArray::int32,$x->dtype());
        } else {
            $this->assertEquals(
                NDArray::int32,$x->dtype());
        }
        $this->assertEquals(
            [100],$x->shape());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
    }

    public function testSlice()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 3D
        $x = $la->array([
            [[0,1,2],
             [3,4,5],
             [6,7,8],
             [9,10,11]],
            [[12,13,14],
             [15,16,17],
             [18,19,20],
             [21,22,23]],
        ]);
        $this->assertEquals(3,$x->ndim());
        $y = $la->slice(
            $x,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [[3,4,5],
             [6,7,8],],
            [[15,16,17],
             [18,19,20],],
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[0,1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[3,4,5],],
            [[15,16,17],]
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[0,-1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[9,10,11],],
            [[21,22,23],]
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[1],
            $size=[1]
            );
        $this->assertEquals([
            [[12,13,14],
             [15,16,17],
             [18,19,20],
             [21,22,23],],
        ],$y->toArray());

        // 2D
        $x = $la->array($mo->arange(8,null,null,dtype:NDArray::float32)->reshape([2,4]));
        $this->assertEquals(2,$x->ndim());
        $x = $la->array([
            [0,1,2,3],
            [4,5,6,7],
        ]);
        $y = $la->slice(
            $x,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [1,2],
            [5,6]
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[0,0],
            $size=[2,4]
            );
        $this->assertEquals([
            [0,1,2,3],
            [4,5,6,7],
        ],$y->toArray());

        // 4D
        $x = $la->array([
            [[[0,1,2],
              [3,4,5]],
             [[6,7,8],
              [9,10,11]]],
            [[[12,13,14],
              [15,16,17]],
             [[18,19,20],
              [21,22,23]]],
        ]);
        $this->assertEquals(4,$x->ndim());
        $y = $la->slice(
            $x,
            $start=[1,1,1],
            $size=[-1,-1,-1]);
        $this->assertEquals([
            [[[21,22,23]]],
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[1,1],
            $size=[-1,-1]);
        $this->assertEquals([
            [[[18,19,20],
              [21,22,23]]],
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[1],
            $size=[-1]);
        $this->assertEquals([
            [[[12,13,14],
              [15,16,17]],
             [[18,19,20],
              [21,22,23]]],
        ],$y->toArray());
    }

    public function testStick()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->array($mo->arange(12,null,null,dtype:NDArray::float32)->reshape([2,2,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,1,2],
             [3,4,5],
             [0,0,0]],
            [[0,0,0],
             [6,7,8],
             [9,10,11],
             [0,0,0]],
        ],$y->toArray());

        $x = $la->array($mo->arange(6,null,null,dtype:NDArray::float32)->reshape([2,1,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[0,1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,1,2],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [3,4,5],
             [0,0,0],
             [0,0,0]],
        ],$y->toArray());

        $x = $la->array($mo->arange(6,null,null,dtype:NDArray::float32)->reshape([2,1,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[0,-1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,0,0],
             [0,0,0],
             [0,1,2]],
            [[0,0,0],
             [0,0,0],
             [0,0,0],
             [3,4,5]],
        ],$y->toArray());

        $x = $la->array($mo->arange(12,null,null,dtype:NDArray::float32)->reshape([1,4,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[1],
            $size=[1]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,0,0],
             [0,0,0],
             [0,0,0]],
            [[0,1,2],
             [3,4,5],
             [6,7,8],
             [9,10,11]],
        ],$y->toArray());

        $x = $la->array($mo->arange(4,null,null,dtype:NDArray::float32)->reshape([2,2]));
        $y = $la->array($mo->zeros([2,4]));
        $la->stick(
            $x,
            $y,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [0,0,1,0],
            [0,2,3,0],
        ],$y->toArray());

        // 4D
        $x = $la->array([
            [[[0,1,2],
              [3,4,5]],
             [[6,7,8],
              [9,10,11]]],
            [[[12,13,14],
              [15,16,17]],
             [[18,19,20],
              [21,22,23]]],
        ]);
        $this->assertEquals(4,$x->ndim());
        $this->assertEquals([2,2,2,3],$x->shape());
        $y = $la->array($mo->zeros([2,2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[ 0, 0, 1],
            $size= [-1,-1, 2]
            );
        $this->assertEquals([
            [[[0,0,0],
              [0,1,2],
              [3,4,5],
              [0,0,0]],
             [[0,0,0],
              [6,7,8],
              [9,10,11],
              [0,0,0]]],
            [[[0,0,0],
              [12,13,14],
              [15,16,17],
              [0,0,0]],
             [[0,0,0],
              [18,19,20],
              [21,22,23],
              [0,0,0]]],
        ],$y->toArray());
    }

    public function testStack()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $a = $la->array($mo->arange(6,0,null,dtype:NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(6,6,null,dtype:NDArray::float32)->reshape([2,3]));
        $y = $la->stack(
            [$a,$b],
            axis:0
            );
        $this->assertEquals([
            [[0,1,2],
             [3,4,5]],
            [[6,7,8],
             [9,10,11]],
        ],$y->toArray());

        $a = $la->array($mo->arange(6,0,null,dtype:NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(6,6,null,dtype:NDArray::float32)->reshape([2,3]));
        $y = $la->stack(
            [$a,$b],
            axis:1
            );
        $this->assertEquals([
            [[0,1,2],
             [6,7,8]],
            [[3,4,5],
             [9,10,11]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,0,null,dtype:NDArray::float32)->reshape([2, 2,3]));
        $b = $la->array($mo->arange(12,12,null,dtype:NDArray::float32)->reshape([2,2,3]));
        $y = $la->stack(
            [$a,$b],
            axis:0
            );
        $this->assertEquals([
           [[[0,1,2],
             [3,4,5]],
            [[6,7,8],
             [9,10,11]]],
           [[[12,13,14],
             [15,16,17]],
            [[18,19,20],
             [21,22,23]]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,0,null,dtype:NDArray::float32)->reshape([2, 2,3]));
        $b = $la->array($mo->arange(12,12,null,dtype:NDArray::float32)->reshape([2,2,3]));
        $y = $la->stack(
            [$a,$b],
            axis:1
            );
        $this->assertEquals([
           [[[0,1,2],
             [3,4,5]],
            [[12,13,14],
             [15,16,17]]],
           [[[6,7,8],
             [9,10,11]],
            [[18,19,20],
             [21,22,23]]],
        ],$y->toArray());

        // 4D
        $a = $la->array($mo->arange(24, 0,null,dtype:NDArray::float32)->reshape([2,2,2,3]));
        $b = $la->array($mo->arange(24,24,null,dtype:NDArray::float32)->reshape([2,2,2,3]));
        $y = $la->stack(
            [$a,$b],
            axis:2
            );
        $this->assertEquals([
             [[[[ 0,  1,  2],
                [ 3,  4,  5]],
               [[24, 25, 26],
                [27, 28, 29]]],
              [[[ 6,  7,  8],
                [ 9, 10, 11]],
               [[30, 31, 32],
                [33, 34, 35]]]],
             [[[[12, 13, 14],
                [15, 16, 17]],
               [[36, 37, 38],
                [39, 40, 41]]],
              [[[18, 19, 20],
                [21, 22, 23]],
               [[42, 43, 44],
                [45, 46, 47]]]]
        ],$y->toArray());
    }

    public function testAnytypeSlice()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        if($la->fp64()) {
            $dtypes = [NDArray::float32,NDArray::float64,NDArray::uint8,NDArray::int32,NDArray::int64];
        } else {
            $dtypes = [NDArray::float32,NDArray::uint8,NDArray::int32,NDArray::int64];
        }
        foreach($dtypes as $dtype) {
            // forward slice
            $x = $la->array($mo->arange(24,null,null,dtype:$dtype)->reshape([2,4,3]));
            $y = $la->slice(
                $x,
                $start=[0,1],
                $size=[-1,2]
                );
            $this->assertEquals([
                [[3,4,5],
                 [6,7,8],],
                [[15,16,17],
                 [18,19,20],],
            ],$y->toArray());

            // reverse slice
            $x = $la->array($mo->arange(12,null,null,dtype:$dtype)->reshape([2,2,3]));
            $y = $la->array($mo->zeros([2,4,3],dtype:$dtype));
            $la->stick(
                $x,
                $y,
                $start=[0,1],
                $size=[-1,2]
                );
            $this->assertEquals([
                [[0,0,0],
                 [0,1,2],
                 [3,4,5],
                 [0,0,0]],
                [[0,0,0],
                 [6,7,8],
                 [9,10,11],
                 [0,0,0]],
            ],$y->toArray());

            // reverse and add
            // $Y = $la->array([
            //     [[1,2,3],[1,2,3]],
            //     [[4,5,6],[4,5,6]],
            // ],$dtype);
            // $X = $la->reduceSumRepeated($Y);
            // $this->assertEquals([2,2,3],$Y->shape());
            // $this->assertEquals([2,3],$X->shape());
            // $this->assertEquals([
            //     [[1,2,3],[1,2,3]],
            //     [[4,5,6],[4,5,6]],
            // ],$Y->toArray());
            // $this->assertEquals([
            //     [2,4,6],
            //     [8,10,12]
            // ],$X->toArray());

        }
    }

    public function testConcat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $a = $la->array($mo->arange(6,$start=0,null,dtype:NDArray::float32)->reshape([3,2]));
        $b = $la->array($mo->arange(4,$start=6,null,dtype:NDArray::float32)->reshape([2,2]));
        $y = $la->concat(
            [$a,$b],
            axis:0
            );
        $this->assertEquals([
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
        ],$y->toArray());

        $a = $la->array($mo->arange(6,$start=0,null,dtype:NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(4,$start=6,null,dtype:NDArray::float32)->reshape([2,2]));
        $y = $la->concat(
            [$a,$b],
            axis:1
            );
        $this->assertEquals([
            [0,1,2,6,7],
            [3,4,5,8,9],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,$start=0,null,dtype:NDArray::float32)->reshape([3,2,2]));
        $b = $la->array($mo->arange(8,$start=12,null,dtype:NDArray::float32)->reshape([2,2,2]));
        $y = $la->concat(
            [$a,$b],
            axis:0
            );
        $this->assertEquals([
            [[0,1],[2,3]],
            [[4,5],[6,7]],
            [[8,9],[10,11]],
            [[12,13],[14,15]],
            [[16,17],[18,19]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,$start=0,null,dtype:NDArray::float32)->reshape([2,3,2]));
        $b = $la->array($mo->arange(8,$start=12,null,dtype:NDArray::float32)->reshape([2,2,2]));
        $y = $la->concat(
            [$a,$b],
            axis:1
            );
        $this->assertEquals([
            [[0,1],
             [2,3],
             [4,5],
             [12,13],
             [14,15]],
            [[6,7],
             [8,9],
             [10,11],
             [16,17],[18,19]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,$start=0,null,dtype:NDArray::float32)->reshape([2,2,3]));
        $b = $la->array($mo->arange(8,$start=12,null,dtype:NDArray::float32)->reshape([2,2,2]));
        $y = $la->concat(
            [$a,$b],
            axis:2
            );
        $this->assertEquals([
            [[0,1,2,12,13],
             [3,4,5,14,15]],
            [[6,7,8,16,17],
             [9,10,11,18,19]],
        ],$y->toArray());

        $y = $la->concat(
            [$a,$b],
            axis:-1
            );
        $this->assertEquals([
            [[0,1,2,12,13],
             [3,4,5,14,15]],
            [[6,7,8,16,17],
             [9,10,11,18,19]],
        ],$y->toArray());
    }

    public function testRepeatNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,axis:1);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());

        // 1 time
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->repeat($X,$repeats=1,axis:1);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,1,3],$Y->shape());
        $this->assertEquals(
            [[[1,2,3]],[[4,5,6]]]
        ,$Y->toArray());

        //
        $X = $la->array([
            [[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]]
        ]);
        $Y = $la->repeat($X,$repeats=4,axis:1);
        $this->assertEquals([
            [[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]]
        ],$X->toArray());
        $this->assertEquals([2,2,3],$X->shape());
        $this->assertEquals([2,4,2,3],$Y->shape());
        $this->assertEquals([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ],$Y->toArray());

        // axis = 0
        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,axis:0);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[4,5,6]],
            [[1,2,3],[4,5,6]],
        ],$Y->toArray());

        // axis = 0
        // Y := X (duplicate 1D)
        $X = $la->array([1,2,3]);
        $Y = $la->repeat($X,$repeats=2,axis:0);
        $this->assertEquals([3],$X->shape());
        $this->assertEquals([2,3],$Y->shape());
        $this->assertEquals([1,2,3],$X->toArray());
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$Y->toArray());

        // axis = 1
        // Y := X (duplicate 1D)
        $X = $la->array([1,2,3]);
        $Y = $la->repeat($X,$repeats=2,axis:1);
        $this->assertEquals([3],$X->shape());
        $this->assertEquals([3,2],$Y->shape());
        $this->assertEquals([1,2,3],$X->toArray());
        $this->assertEquals([
            [1,1],
            [2,2],
            [3,3],
        ],$Y->toArray());

        // axis = NULL
        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,axis:null);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([12],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            1,2,3,4,5,6,
            1,2,3,4,5,6,
        ],$Y->toArray());

        // axis = -1
        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,axis:-1);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());

        // keepdims
        $X = $la->ones($la->alloc([2,3,4]));
        $Y = $la->repeat($X,$repeats=2,axis:0,keepdims:true);
        $this->assertEquals([4,3,4],$Y->shape());
        $Z = $la->repeat($X,$repeats=2,axis:0);
        $this->assertEquals($Z->reshape([4*3*4])->toArray(),$Y->reshape([4*3*4])->toArray());

        $X = $la->ones($la->alloc([2,3,4]));
        $Y = $la->repeat($X,$repeats=2,axis:1,keepdims:true);
        $this->assertEquals([2,6,4],$Y->shape());
        $Z = $la->repeat($X,$repeats=2,axis:1);
        $this->assertEquals($Z->reshape([2*6*4])->toArray(),$Y->reshape([2*6*4])->toArray());

        $X = $la->ones($la->alloc([2,3,4]));
        $Y = $la->repeat($X,$repeats=2,axis:2,keepdims:true);
        $this->assertEquals([2,3,8],$Y->shape());
        $Z = $la->repeat($X,$repeats=2,axis:1);
        $this->assertEquals($Z->reshape([2*3*8])->toArray(),$Y->reshape([2*3*8])->toArray());

        $X = $la->ones($la->alloc([2,3,4]));
        $Y = $la->repeat($X,$repeats=2,axis:null,keepdims:true);
        $this->assertEquals([48],$Y->shape());
        $Z = $la->repeat($X,$repeats=2,axis:null);
        $this->assertEquals($Z->reshape([48])->toArray(),$Y->reshape([48])->toArray());
    }

    public function testReduceSum3d()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X (sum 2 times)
        $Y = $la->array([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ]);
        $X = $la->reduceSum($Y,axis:1);
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());
        $this->assertEquals([
            [2,4,6],
            [8,10,12]
        ],$X->toArray());

        // 1 time
        $Y = $la->array([
            [[1,2,3]],
            [[4,5,6]]
        ]);
        $X = $la->reduceSum($Y,axis:1);
        $this->assertEquals([2,1,3],$Y->shape());
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3]],
            [[4,5,6]]
        ],$Y->toArray());

        $Y = $la->array([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ]);
        $X = $la->reduceSum($Y,axis:1);
        $this->assertEquals([2,4,2,3],$Y->shape());
        $this->assertEquals([2,2,3],$X->shape());
        $this->assertEquals([
            [[4,8,12],[16,20,24]],
            [[28,32,36],[40,44,48]]
        ],$X->toArray());
        $this->assertEquals([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ],$Y->toArray());
    }

    public function testSplit()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->array([
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            axis:0
        );
        $a = $la->array($mo->arange(6,$start=0,null,dtype:NDArray::float32)->reshape([3,2]));
        $b = $la->array($mo->arange(4,$start=6,null,dtype:NDArray::float32)->reshape([2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [0,1,2,6,7],
            [3,4,5,8,9],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            axis:1
            );
        $a = $la->array($mo->arange(6,$start=0,null,dtype:NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(4,$start=6,null,dtype:NDArray::float32)->reshape([2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [[0,1],[2,3]],
            [[4,5],[6,7]],
            [[8,9],[10,11]],
            [[12,13],[14,15]],
            [[16,17],[18,19]],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            axis:0
            );
        $a = $la->array($mo->arange(12,$start=0,null,dtype:NDArray::float32)->reshape([3,2,2]));
        $b = $la->array($mo->arange(8,$start=12,null,dtype:NDArray::float32)->reshape([2,2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [[0,1],
             [2,3],
             [4,5],
             [12,13],
             [14,15]],
            [[6,7],
             [8,9],
             [10,11],
             [16,17],[18,19]],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            axis:1
            );
        $a = $la->array($mo->arange(12,$start=0,null,dtype:NDArray::float32)->reshape([2,3,2]));
        $b = $la->array($mo->arange(8,$start=12,null,dtype:NDArray::float32)->reshape([2,2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [[0,1,2,12,13],
             [3,4,5,14,15]],
            [[6,7,8,16,17],
             [9,10,11,18,19]],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            axis:2
            );
        $a = $la->array($mo->arange(12,$start=0,null,dtype:NDArray::float32)->reshape([2,2,3]));
        $b = $la->array($mo->arange(8,$start=12,null,dtype:NDArray::float32)->reshape([2,2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $y = $la->split(
            $x,
            [3,2],
            axis:-1
            );
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());
    }

    public function testTranspose()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D
        $a = $la->array([1,2,3,4,5,6],dtype:NDArray::float32);
        $b = $la->transpose($a);
        $this->assertEquals(
            [1,2,3,4,5,6],
            $b->toArray()
        );
        $this->assertEquals([6],$a->shape());
        $this->assertEquals([6],$b->shape());

        // 2D float
        $a = $la->array([
            [0,1,2],
            [3,4,5],
        ]);
        $b = $la->transpose($a);
        $this->assertEquals([
            [0,3],
            [1,4],
            [2,5]
        ],$b->toArray());

        // 2D int
        $a = $la->array([
            [0,1,2],
            [3,4,5],
        ],dtype:NDArray::int32);
        $b = $la->transpose($a);
        $this->assertEquals([
            [0,3],
            [1,4],
            [2,5]
        ],$b->toArray());

        // 3D
        $a = $la->array(
            [[[ 0,  1,  2],
              [ 3,  4,  5]],
             [[ 6,  7,  8],
              [ 9, 10, 11]],
             [[12, 13, 14],
              [15, 16, 17]],
             [[18, 19, 20],
              [21, 22, 23]]],
              dtype:NDArray::float32);
        $b = $la->transpose($a);
        $this->assertEquals(
            [[[ 0,  6, 12, 18],
              [ 3,  9, 15, 21]],
             [[ 1,  7, 13, 19],
              [ 4, 10, 16, 22]],
             [[ 2,  8, 14, 20],
              [ 5, 11, 17, 23]]],
            $b->toArray()
        );
        $this->assertEquals([4,2,3],$a->shape());
        $this->assertEquals([3,2,4],$b->shape());

        // 4D  use full spec of algorism
        $a = $la->array(
        [[[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15],
         [16, 17]],

        [[18, 19],
         [20, 21],
         [22, 23]]],


       [[[24, 25],
         [26, 27],
         [28, 29]],

        [[30, 31],
         [32, 33],
         [34, 35]],

        [[36, 37],
         [38, 39],
         [40, 41]],

        [[42, 43],
         [44, 45],
         [46, 47]]]],
         dtype:NDArray::float32);
        $b = $la->transpose($a);
        $this->assertEquals(
            [[[[ 0, 24],
         [ 6, 30],
         [12, 36],
         [18, 42]],

        [[ 2, 26],
         [ 8, 32],
         [14, 38],
         [20, 44]],

        [[ 4, 28],
         [10, 34],
         [16, 40],
         [22, 46]]],


       [[[ 1, 25],
         [ 7, 31],
         [13, 37],
         [19, 43]],

        [[ 3, 27],
         [ 9, 33],
         [15, 39],
         [21, 45]],

        [[ 5, 29],
         [11, 35],
         [17, 41],
         [23, 47]]]],
            $b->toArray()
        );
        $this->assertEquals([2,4,3,2],$a->shape());
        $this->assertEquals([2,3,4,2],$b->shape());

        // with perm
        $a = $la->array(
            [[[ 0,  1,  2],
              [ 3,  4,  5]],
             [[ 6,  7,  8],
              [ 9, 10, 11]],
             [[12, 13, 14],
              [15, 16, 17]],
             [[18, 19, 20],
              [21, 22, 23]]],
              dtype:NDArray::float32);
        $b = $la->transpose(
            $a,$perm=[2,0,1],
        );
        $this->assertEquals(
            [[[ 0,  3],
              [ 6,  9],
              [12, 15],
              [18, 21]],
             [[ 1,  4],
              [ 7, 10],
              [13, 16],
              [19, 22]],
             [[ 2,  5],
              [ 8, 11],
              [14, 17],
              [20, 23]]],
            $b->toArray()
        );
        $this->assertEquals([4,2,3],$a->shape());
        $this->assertEquals([3,4,2],$b->shape());

    }

    public function testBandpartFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->ones($la->alloc([2,5,5]));
        $la->bandpart($a,0,-1);
        $this->assertEquals([
            [[1,1,1,1,1],
             [0,1,1,1,1],
             [0,0,1,1,1],
             [0,0,0,1,1],
             [0,0,0,0,1]],
            [[1,1,1,1,1],
             [0,1,1,1,1],
             [0,0,1,1,1],
             [0,0,0,1,1],
             [0,0,0,0,1]],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5]));
        $la->bandpart($a,-1,0);
        $this->assertEquals([
            [1,0,0,0,0],
            [1,1,0,0,0],
            [1,1,1,0,0],
            [1,1,1,1,0],
            [1,1,1,1,1],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5]));
        $la->bandpart($a,0,0);
        $this->assertEquals([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5]));
        $la->bandpart($a,0,1);
        $this->assertEquals([
            [1,1,0,0,0],
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,1],
            [0,0,0,0,1],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5]));
        $la->bandpart($a,1,0);
        $this->assertEquals([
            [1,0,0,0,0],
            [1,1,0,0,0],
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,1],
        ],$a->toArray());
    }

    public function testBandpartBool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->ones($la->alloc([2,5,5],dtype:NDArray::bool));
        $la->bandpart($a,0,-1);
        $this->assertEquals([
            [[true, true, true, true, true],
             [false,true, true, true, true],
             [false,false,true, true, true],
             [false,false,false,true, true],
             [false,false,false,false,true]],
            [[true, true, true, true, true],
             [false,true, true, true, true],
             [false,false,true, true, true],
             [false,false,false,true, true],
             [false,false,false,false,true]],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5],dtype:NDArray::bool));
        $la->bandpart($a,-1,0);
        $this->assertEquals([
            [true, false,false,false,false],
            [true, true, false,false,false],
            [true, true, true, false,false],
            [true, true, true, true, false],
            [true, true, true, true, true ],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5],dtype:NDArray::bool));
        $la->bandpart($a,0,0);
        $this->assertEquals([
            [true, false,false,false,false],
            [false,true, false,false,false],
            [false,false,true, false,false],
            [false,false,false,true, false],
            [false,false,false,false,true ],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5],dtype:NDArray::bool));
        $la->bandpart($a,0,1);
        $this->assertEquals([
            [true, true, false,false,false],
            [false,true, true, false,false],
            [false,false,true, true, false],
            [false,false,false,true, true ],
            [false,false,false,false,true ],
        ],$a->toArray());

        $a = $la->ones($la->alloc([5,5],dtype:NDArray::bool));
        $la->bandpart($a,1,0);
        $this->assertEquals([
            [true, false,false,false,false],
            [true, true, false,false,false],
            [false,true, true, false,false],
            [false,false,true, true, false],
            [false,false,false,true, true ],
        ],$a->toArray());
    }

    public function testImagecopy()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [[0],[1],[2]],
            [[3],[4],[5]],
            [[6],[7],[8]],
        ]);
        $b = $la->imagecopy($a,null,null,
            $heightShift=1
        );
        $this->assertEquals([
            [[0],[1],[2]],
            [[0],[1],[2]],
            [[3],[4],[5]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1
        );
        $this->assertEquals([
            [[3],[4],[5]],
            [[6],[7],[8]],
            [[6],[7],[8]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=1
        );
        $this->assertEquals([
            [[0],[0],[1]],
            [[3],[3],[4]],
            [[6],[6],[7]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=-1
        );
        $this->assertEquals([
            [[1],[2],[2]],
            [[4],[5],[5]],
            [[7],[8],[8]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=1
        );
        $this->assertEquals([
            [[0],[0],[1]],
            [[0],[0],[1]],
            [[3],[3],[4]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=-1
        );
        $this->assertEquals([
            [[4],[5],[5]],
            [[7],[8],[8]],
            [[7],[8],[8]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=null
        );
        $this->assertEquals([
            [[6],[7],[8]],
            [[3],[4],[5]],
            [[0],[1],[2]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=null,
            $verticalFlip=null,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[2],[1],[0]],
            [[5],[4],[3]],
            [[8],[7],[6]],
        ],$b->toArray());

        $a = $la->array([
            [[0], [1], [2], [3] ],
            [[4], [5], [6], [7] ],
            [[8], [9], [10],[11]],
            [[12],[13],[14],[15]],
        ]);
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=null
        );
        $this->assertEquals([
            [[12],[13],[14],[15]],
            [[12],[13],[14],[15]],
            [[8],[9],[10],[11]],
            [[4],[5],[6],[7]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=null
        );
        $this->assertEquals([
            [[8], [9], [10],[11]],
            [[4], [5], [6], [7] ],
            [[0], [1], [2], [3] ],
            [[0], [1], [2], [3] ],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=1,
            $verticalFlip=null,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[3] ,[3] ,[2], [1] ],
            [[7] ,[7] ,[6], [5] ],
            [[11],[11],[10],[9] ],
            [[15],[15],[14],[13]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=-1,
            $verticalFlip=null,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[2 ],[1 ],[0] ,[0] ],
            [[6 ],[5 ],[4] ,[4] ],
            [[10],[9 ],[8] ,[8] ],
            [[14],[13],[12],[12]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[14],[13],[12]],
            [[11],[10],[9] ,[8] ],
            [[7] ,[6] ,[5] ,[4] ],
            [[3] ,[2] ,[1] ,[0] ],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[14],[13],[12]],
            [[15],[14],[13],[12]],
            [[11],[10],[9] ,[8]],
            [[7] ,[6], [5] ,[4]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[11],[10],[9] ,[8]],
            [[7] ,[6], [5] ,[4]],
            [[3] ,[2], [1] ,[0]],
            [[3] ,[2], [1] ,[0]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[15],[14],[13]],
            [[11],[11],[10],[9 ]],
            [[7 ],[7 ],[6 ],[5 ]],
            [[3 ],[3 ],[2 ],[1 ]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=-1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[14],[13],[12],[12]],
            [[10],[9 ],[8 ],[8 ]],
            [[6 ],[5 ],[4 ],[4 ]],
            [[2 ],[1 ],[0 ],[0 ]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[15],[14],[13]],
            [[15],[15],[14],[13]],
            [[11],[11],[10],[9] ],
            [[7] ,[7] ,[6] ,[5] ],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=-1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[10],[9] ,[8] ,[8] ],
            [[6] ,[5] ,[4] ,[4] ],
            [[2] ,[1] ,[0] ,[0] ],
            [[2] ,[1] ,[0] ,[0] ],
        ],$b->toArray());

        $a = $la->array([
            [[1,2,3],
             [1,2,3]],
            [[4,5,6],
             [4,5,6]],
            [[7,8,9],
             [7,8,9]],
        ]);
        $this->assertEquals([3,2,3],$a->shape());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[1,2,3],
             [1,2,3]],
            [[1,2,3],
             [1,2,3]],
            [[4,5,6],
             [4,5,6]],
        ],$b->toArray());

        // flip rgb
        $a = $la->array([
            [[1,2,3],
             [1,2,3]],
            [[4,5,6],
             [4,5,6]],
            [[7,8,9],
             [7,8,9]],
        ]);
        $this->assertEquals([3,2,3],$a->shape());
        $b = $la->imagecopy($a,null,null,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false,
            $rgbFlip=true
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[3,2,1],
             [3,2,1]],
            [[6,5,4],
             [6,5,4]],
            [[9,8,7],
             [9,8,7]],
        ],$b->toArray());

        // flip rgb with alpha
        $a = $la->array([
            [[1,2,3,4],
             [1,2,3,4]],
            [[4,5,6,7],
             [4,5,6,7]],
            [[7,8,9,10],
             [7,8,9,10]],
        ]);
        $this->assertEquals([3,2,4],$a->shape());
        $b = $la->imagecopy($a,null,null,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false,
            $rgbFlip=true
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[3,2,1,4],
             [3,2,1,4]],
            [[6,5,4,7],
             [6,5,4,7]],
            [[9,8,7,10],
             [9,8,7,10]],
        ],$b->toArray());
    }

    public function testImagecopychannelsfirst()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [[1,2,3],
             [4,5,6]],
            [[11,12,13],
             [14,15,16]],
            [[21,22,23],
             [24,25,26]],
        ]);
        $this->assertEquals([3,2,3],$a->shape());
        $b = $la->imagecopy($a,null,true,
            $heightShift=1,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false
        );
        $this->assertEquals([
            [[1,2,3],
             [1,2,3]],
            [[11,12,13],
             [11,12,13]],
            [[21,22,23],
             [21,22,23]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,true,
            $heightShift=0,
            $widthShift=1,
            $verticalFlip=false,
            $horizontalFlip=false
        );
        $this->assertEquals([
            [[1,1,2],
             [4,4,5]],
            [[11,11,12],
             [14,14,15]],
            [[21,21,22],
             [24,24,25]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,true,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=true,
            $horizontalFlip=false
        );
        $this->assertEquals([
            [[4,5,6],
             [1,2,3]],
            [[14,15,16],
             [11,12,13]],
            [[24,25,26],
             [21,22,23]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,true,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=true
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[3,2,1],
             [6,5,4]],
            [[13,12,11],
             [16,15,14]],
            [[23,22,21],
             [26,25,24]],
        ],$b->toArray());
    }

    public function testfillNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([2,3],NDArray::float32);
        $b = $la->fill(123,$x);
        $this->assertEquals([
            [123,123,123],
            [123,123,123],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::int64);
        $b = $la->fill(456,$x);
        $this->assertEquals([
            [456,456,456],
            [456,456,456],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::float32);
        $value = $mo->array(345); // value is not OpenCL buffer
        $b = $la->fill($value,$x);
        $this->assertEquals([
            [345,345,345],
            [345,345,345],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::int8);
        $b = $la->fill(123,$x);
        $this->assertEquals([
            [123,123,123],
            [123,123,123],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::uint8);
        $b = $la->fill(234,$x);
        $this->assertEquals([
            [234,234,234],
            [234,234,234],
        ],$b->toArray());

        $x = $la->alloc([2,3],dtype:NDArray::bool);
        $b = $la->fill(true,$x);
        $this->assertEquals([
            [true,true,true],
            [true,true,true],
        ],$b->toArray());
        $b = $la->fill(false,$x);
        $this->assertEquals([
            [false,false,false],
            [false,false,false],
        ],$b->toArray());

        $x = $la->alloc([2,3],dtype:NDArray::bool);
        $value = $mo->array(true,dtype:NDArray::bool); // value is not OpenCL buffer
        $b = $la->fill($value,$x);
        $this->assertEquals([
            [true,true,true],
            [true,true,true],
        ],$b->toArray());
        $value = $mo->array(false,dtype:NDArray::bool); // value is not OpenCL buffer
        $b = $la->fill($value,$x);
        $this->assertEquals([
            [false,false,false],
            [false,false,false],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::complex64);
        $b = $la->fill(C(12,i:3),$x);
        $this->assertEquals([
            [C(12,i:3),C(12,i:3),C(12,i:3)],
            [C(12,i:3),C(12,i:3),C(12,i:3)],
        ],$b->toArray());
    }

    public function testSearchsorted()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $A = $mo->array([0.1,0.3,0.5,0.7,0.9]);
        $A = $la->array($A);
        $X = $mo->array([0.0,0.5,1.0]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0,2,5],
            $Y->toArray()
        );

        // right=true
        $A = $mo->array([0.1,0.3,0.5,0.7,0.9]);
        $A = $la->array($A);
        $X = $mo->array([0.0,0.5,1.0]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [0,3,5],
            $Y->toArray()
        );

        // individual mode
        $A = $mo->array([
            [1,   3,  5,   7,   9],
            [1,   2,  3,   4,   5],
            [0, 100, 20, 300, 400]
        ]);
        $A = $la->array($A);
        $X = $mo->array([0, 5, 10]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0, 4, 1],
            $Y->toArray()
        );

        // individual mode & right=true
        $A = $mo->array([
            [1,   3,  5,   7,   9],
            [1,   2,  3,   4,   5],
            [0, 100, 20, 300, 400]
        ]);
        $A = $la->array($A);
        $X = $mo->array([0, 5, 10]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [0, 5, 1],
            $Y->toArray()
        );

        // after nan2num and cumsum
        //  nan nan 5 nan 4
        $A = $mo->array([NAN,NAN,0.5,NAN,0.5]);
        $A = $la->array($A);
        $A = $la->nan2num($A,0);
        $total = $la->sum($A);
        $this->assertEquals(1.0,$total);
        $size = $A->size();
        $this->assertEquals(5,$size);
        $A = $la->cumsum($A[R(0,$size-1)]);
        $this->assertEquals([0.0,0.0,0.5,0.5],$A->toArray());
        $X = $mo->array([0.0,0.4,0.6,$total]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [2,2,4,4],
            $Y->toArray()
        );

        // right=true
        $A = $mo->array([0.5,1.0,1.0]);
        $A = $la->array($A);
        $X = $mo->array([0.9]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [1],
            $Y->toArray()
        );


        // nan data
        $A = $mo->array([1,3,5,7,9]);
        $A = $la->array($A);
        $X = $mo->array([NAN,5,10]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0,2,5],
            $Y->toArray()
        );

        // nan seq
        $A = $mo->array([0.1,0.3,NAN,0.7,0.9]);
        $A = $la->array($A);
        $X = $mo->array([0.0,0.5,1.0]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0,2,2],
            $Y->toArray()
        );
    }

    public function testcumsumNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $mo->array([1,2,1,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X);
        $this->assertEquals(
            [1,3,4,6],
            $Y->toArray()
        );

        $X = $mo->array([1,2,3,4]);
        $X = $la->array($X);
        $Y = $la->cumsum($X);
        $this->assertEquals(
            [1,3,6,10],
            $Y->toArray()
        );

        // exclusive=true
        $X = $mo->array([1,2,1,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,exclusive:true);
        $this->assertEquals(
            [0,1,3,4],
            $Y->toArray()
        );

        $X = $mo->array([1,2,3,4]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,exclusive:true);
        $this->assertEquals(
            [0, 1, 3, 6],
            $Y->toArray()
        );

        // reverse=true
        $X = $mo->array([1,2,1,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,reverse:true);
        $this->assertEquals(
            [6,5,3,2],
            $Y->toArray()
        );

        $X = $mo->array([1,2,3,4]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,reverse:true);
        $this->assertEquals(
            [10,9,7,4],
            $Y->toArray()
        );

        // exclusive=true & reverse=true
        $X = $mo->array([1,2,3,4]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,exclusive:true,reverse:true);
        $this->assertEquals(
            [9, 7, 4, 0],
            $Y->toArray()
        );

        // nan data
        $X = $mo->array([1,2,NAN,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X);
        $Y = $la->toNDArray($Y);
        $this->assertEquals(1.0,$Y[0]);
        $this->assertEquals(3.0,$Y[1]);
        $this->assertTrue(is_nan($Y[2]));
        $this->assertTrue(is_nan($Y[3]));

        // nan data with reverse
        $X = $mo->array([1,2,NAN,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,null,reverse:true);
        $Y = $la->toNDArray($Y);
        $this->assertTrue(is_nan($Y[0]));
        $this->assertTrue(is_nan($Y[1]));
        $this->assertTrue(is_nan($Y[2]));
        $this->assertEquals(2.0,$Y[3]);
        //$this->assertEquals(3.0,$Y[2]);
        //$this->assertEquals(1.0,$Y[3]);
    }

    public function testcumsumWithAxis()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $mo->array([
            [[ 1, 2, 3],[ 4, 5, 6],[ 7, 8, 9]],
            [[11,12,13],[14,15,16],[17,18,19]],
            [[21,22,23],[24,25,26],[27,28,29]],
        ]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,axis:0);
        $this->assertEquals([
            [[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9]],
    
            [[12, 14, 16],
             [18, 20, 22],
             [24, 26, 28]],
    
            [[33, 36, 39],
             [42, 45, 48],
             [51, 54, 57]]
        ],$Y->toArray());

        $Y = $la->cumsum($X,axis:1);
        $this->assertEquals([
            [[ 1,  2,  3],
             [ 5,  7,  9],
             [12, 15, 18]],
    
            [[11, 12, 13],
             [25, 27, 29],
             [42, 45, 48]],
    
            [[21, 22, 23],
             [45, 47, 49],
             [72, 75, 78]]
        ],$Y->toArray());

        $Y = $la->cumsum($X,axis:2);
        $this->assertEquals([
            [[ 1,  3,  6],
             [ 4,  9, 15],
             [ 7, 15, 24]],
    
            [[11, 23, 36],
             [14, 29, 45],
             [17, 35, 54]],
    
            [[21, 43, 66],
             [24, 49, 75],
             [27, 55, 84]]
        ],$Y->toArray());

        // exclusive
        $Y = $la->cumsum($X,axis:1,exclusive:true);
        $this->assertEquals([
            [[ 0,  0,  0],
            [ 1,  2,  3],
            [ 5,  7,  9]],
    
           [[ 0,  0,  0],
            [11, 12, 13],
            [25, 27, 29]],
    
           [[ 0,  0,  0],
            [21, 22, 23],
            [45, 47, 49]]
        ],$Y->toArray());

        // reverse
        $Y = $la->cumsum($X,axis:1,reverse:true);
        $this->assertEquals([
            [[12, 15, 18],
            [11, 13, 15],
            [ 7,  8,  9]],
    
           [[42, 45, 48],
            [31, 33, 35],
            [17, 18, 19]],
    
           [[72, 75, 78],
            [51, 53, 55],
            [27, 28, 29]]
        ],$Y->toArray());

        // exclusive & reverse
        $Y = $la->cumsum($X,axis:1,exclusive:true,reverse:true);
        $this->assertEquals([
            [[11, 13, 15],
            [ 7,  8,  9],
            [ 0,  0,  0]],
    
           [[31, 33, 35],
            [17, 18, 19],
            [ 0,  0,  0]],
    
           [[51, 53, 55],
            [27, 28, 29],
            [ 0,  0,  0]]
        ],$Y->toArray());
    }

    public function testNan2num()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := nan2num(X)
        $X = $mo->array([[NAN,2,NAN],[4,NAN,6]]);
        $X = $la->array($X);
        $la->nan2num($X);
        $this->assertEquals(
            [[0,2,0],[4,0,6]],
            $X->toArray()
        );

        $X = $mo->array([[NAN,2,NAN],[4,NAN,6]]);
        $X = $la->array($X);
        $la->nan2num($X,1.0);
        $this->assertEquals(
            [[1,2,1],[4,1,6]],
            $X->toArray()
        );
    }

    public function testIsnan()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := nan2num(X)
        $X = $mo->array([[NAN,2,NAN],[4,NAN,6]]);
        $X = $la->array($X);
        $la->isnan($X);
        $this->assertEquals(
            [[1,0,1],[0,1,0]],
            $X->toArray()
        );
    }

    public function testLinspace()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $la->linspace($start=10,$stop=100,$num=10);
        $this->assertEquals(
            [10,20,30,40,50,60,70,80,90,100],
            $X->toArray()
        );
    }


    public function testSvdFull1()
    {
        if($this->service->serviceLevel()<Service::LV_ADVANCED) {
            $this->markTestSkipped('Unsuppored function without openblas');
            return;
        }

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
        $this->assertEquals([6,5],$a->shape());
        [$u,$s,$vt] = $la->svd($a);
        $this->assertEquals([6,6],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([5,5],$vt->shape());

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
            [-0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [-0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [-0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [-0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [-0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [ 0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ]);
        //$this->assertTrue(false);
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

    public function testSvdFull2()
    {
        if($this->service->serviceLevel()<Service::LV_ADVANCED) {
            $this->markTestSkipped('Unsuppored function without openblas');
            return;
        }
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
        $a = $la->transpose($a);
        $this->assertEquals([5,6],$a->shape());
        [$u,$s,$vt] = $la->svd($a);
        $this->assertEquals([5,5],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([6,6],$vt->shape());

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
            [ 0.25, 0.40, 0.69, 0.37, 0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $correctU = $la->transpose($correctU);

        $correctU = $la->square($correctU);
        $u = $la->square($u);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [ 0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [ 0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [ 0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [ 0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [ 0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [-0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ]);
        $correctVT = $la->transpose($correctVT);
        $correctVT = $la->square($correctVT);
        $vt = $la->square($vt);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
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

    public function testSvdSmallVT()
    {
        if($this->service->serviceLevel()<Service::LV_ADVANCED) {
            $this->markTestSkipped('Unsuppored function without openblas');
            return;
        }
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
        $a = $la->transpose($a);
        [$u,$s,$vt] = $la->svd($a,$full_matrices=false);

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #  echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #  echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $la->array([
            [ 0.25, 0.40, 0.69, 0.37, 0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $correctU = $la->transpose($correctU);
        $correctU = $la->square($correctU);
        $u = $la->square($u);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [ 0.59, 0.26, 0.36, 0.31, 0.23,],
            [ 0.40, 0.24,-0.22,-0.75,-0.36,],
            [ 0.03,-0.60,-0.45, 0.23,-0.31,],
            [ 0.43, 0.24,-0.69, 0.33, 0.16,],
            [ 0.47,-0.35, 0.39, 0.16,-0.52,],
            [-0.29, 0.58,-0.02, 0.38,-0.65,],
        ]);
        $correctVT = $la->transpose($correctVT);
        $correctVT = $la->square($correctVT);
        $vt = $la->square($vt);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
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

    public function testIsFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::int8)));
        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::uint8)));
        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::int32)));
        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::uint32)));
        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::int64)));
        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::uint64)));

        $this->assertTrue($la->isFloat($la->array(1,dtype:NDArray::float32)));
        $this->assertTrue($la->isFloat($la->array(1,dtype:NDArray::float64)));

        $this->assertFalse($la->isFloat($la->array(1,dtype:NDArray::bool)));
    }

    public function testEinsumSimpleNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // cross product 2D
        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum(' ik , kj -> ij ',$x,$y);
        $this->assertEquals([
            [19,22],
            [43,50]
        ],$z->toArray());

        // use cached equation
        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum(' ik , kj -> ij ',$x,$y);
        $this->assertEquals([
            [19,22],
            [43,50]
        ],$z->toArray());

        // transpose output
        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum(' ik , kj -> ji ',$x,$y);
        $this->assertEquals([
            [19,43],
            [22,50]
        ],$z->toArray());

        // cross product 3D x 2D
        $x = $mo->array(range(0, 4*3*2-1))->reshape([4,3,2]);
        $y = $mo->array(range(0, 5*2-1))->reshape([5,2]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum('abc,dc->abd',$x,$y);
        $this->assertEquals([4,3,5],$z->shape());
        $this->assertEquals([
            [[  1,   3,   5,   7,   9],
             [  3,  13,  23,  33,  43],
             [  5,  23,  41,  59,  77]],

            [[  7,  33,  59,  85, 111],
             [  9,  43,  77, 111, 145],
             [ 11,  53,  95, 137, 179]],

            [[ 13,  63, 113, 163, 213],
             [ 15,  73, 131, 189, 247],
             [ 17,  83, 149, 215, 281]],

            [[ 19,  93, 167, 241, 315],
             [ 21, 103, 185, 267, 349],
             [ 23, 113, 203, 293, 383]],
        ],$z->toArray());


        // multi head attention dot product 
        $shapeX = [2,2,2,2,2];
        $shapeY = [2,2,2,2,2];
        $x = $mo->array(range(0, (int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(0, (int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum('afgde,abcde->adbcfg',$x,$y);
        $this->assertEquals([2,2,2,2,2,2],$z->shape());
        $this->assertEquals([
            [[[[[ 1,     5],[   9,   13]],[[   5,   41],[  77,  113]]],
              [[[ 9,    77],[ 145,  213]],[[  13,  113],[ 213,  313]]]],
             [[[[13,    33],[  53,   73]],[[  33,   85],[ 137,  189]]],
              [[[53,   137],[ 221,  305]],[[  73,  189],[ 305,  421]]]]],
            [[[[[545,  677],[ 809,  941]],[[ 677,  841],[1005, 1169]]],
              [[[809, 1005],[1201, 1397]],[[ 941, 1169],[1397, 1625]]]],
             [[[[685,  833],[ 981, 1129]],[[ 833, 1013],[1193, 1373]]],
              [[[981, 1193],[1405, 1617]],[[1129, 1373],[1617, 1861]]]]],
        ],$z->toArray());

        // multi head attention dot product 4D
        $shapeX = [2,2,2,2];
        $shapeY = [2,2,2,2];
        $x = $mo->array(range(0, (int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(0, (int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum('aecd,abcd->acbe',$x,$y);
        $this->assertEquals([2,2,2,2],$z->shape());
        $this->assertEquals([
          [[[  1,   5],
            [  5,  41]],
           [[ 13,  33],
            [ 33,  85]]],
          [[[145, 213],
            [213, 313]],
           [[221, 305],
            [305, 421]]],
        ],$z->toArray());

        // multi head attention combine 4D
        $shapeX = [2,2,2,2];
        $shapeY = [2,2,2,2];
        $x = $mo->array(range(0, (int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(0, (int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum('acbe,aecd->abcd',$x,$y);
        $this->assertEquals([2,2,2,2],$z->shape());
        $this->assertEquals([
          [[[  4,   5],
            [ 38,  47]],
           [[ 12,  17],
            [ 54,  67]]],
          [[[172, 189],
            [302, 327]],
           [[212, 233],
            [350, 379]]],
        ],$z->toArray());

        // share indicator 
        $shapeX = [2,2];
        $shapeY = [2,3];
        $x = $mo->array(range(1, 1+(int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(1, 1+(int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum('aa,ab->ab',$x,$y);
        $this->assertEquals([2,3],$z->shape());
        $this->assertEquals([
            [ 1,  2,  3],
            [16, 20, 24],
        ],$z->toArray());


        $shapeX = [2,2];
        $shapeY = [2,2];
        $x = $mo->array(range(1, 1+(int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(1, 1+(int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        //$this->expectException(InvalidArgumentException::class);
        //$this->expectExceptionMessage('rank of C must be less than 4D or equal.');
        $z = $la->einsum('ab,ab->ab',$x,$y);
        $this->assertEquals([2,2],$z->shape());
        $this->assertEquals([
            [ 1,  4],
            [ 9, 16],
        ],$z->toArray());

    }

    public function testEinsumExplicitNormal()
    {
        $this->markTestSkipped('Not implemented');
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        
        // cross product
        $z = $la->einsum(' ik , kj -> ij ',$x,$y);
        $this->assertEquals([[19,22],[43,50]],$z->toArray());

        // Diagonal Extraction
        $z = $la->einsum(' ii -> i ',$x);
        $this->assertEquals([1,4],$z->toArray());

        // dot product
        $z = $la->einsum(' ik , kj -> ',$x,$y);
        $this->assertEquals(134,$z->toArray());

        // transpose
        $z = $la->einsum(' ij->ji ',$x);
        $this->assertEquals([[1,3],[2,4]],$z->toArray());

        // sum
        $z = $la->einsum(' ik -> ',$x);
        $this->assertEquals(10,$z->toArray());

    }

    public function testEinsumImplicitNormal()
    {
        $this->markTestSkipped('Not implemented');
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        
        ///////////////////////////////
        // multi tensor mode
        ///////////////////////////////
        // cross product
        $z = $la->einsum(' ik,kj ',$x,$y);
        $this->assertEquals([[19,22],[43,50]],$z->toArray());
        $z = $la->einsum(' jk,ki ',$x,$y);
        $this->assertEquals([[19,43],[22,50]],$z->toArray());

        // dot product
        $z = $la->einsum(' ij,ij ',$x,$y);
        $this->assertEquals(70,$z->toArray());

        ///////////////////////////////
        // single tensor mode
        ///////////////////////////////
        $x = $mo->array([[1,2],[3,4]]);
        $x3 = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        // no transpose
        $z = $la->einsum(' ij ',$x);
        $this->assertEquals([[1,2],[3,4]],$z->toArray());
        $z = $la->einsum(' ijk ',$x3);
        $this->assertEquals([[[1,2],[3,4]],[[5,6],[7,8]]],$z->toArray());

        // transpose
        $z = $la->einsum(' ji ',$x);
        $this->assertEquals([[1,3],[2,4]],$z->toArray());
        $z = $la->einsum(' kji ',$x3);
        $this->assertEquals([[[1,5],[3,7]],[[2,6],[4,8]]],$z->toArray());

        // trace
        $z = $la->einsum(' ii ',$x);
        $this->assertEquals(5,$z->toArray());
        $z = $la->einsum(' iii ',$x3);
        $this->assertEquals(9,$z->toArray());

        // some conversions
        $z = $la->einsum(' ijj ',$x3);
        $this->assertEquals([5,13],$z->toArray());
        $z = $la->einsum(' iij ',$x3);
        $this->assertEquals([8,10],$z->toArray());
    }
    public function testEinsumPlaceholderExplicitNormal()
    {
        $this->markTestSkipped('Not implemented');
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        
        // simple placeholder (left)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]);         // x.shape = (2,2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $z = $la->einsum(' ...i->...i',$x);
        $this->assertEquals([[[1,2],[3,4]],[[5,6],[7,8]]],$z->toArray());

        // simple placeholder (right)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]);         // x.shape = (2,2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $z = $la->einsum(' i...->i...',$x);
        $this->assertEquals([[[1,2],[3,4]],[[5,6],[7,8]]],$z->toArray());

        // transpose with blaceholder (left)
        $x = $mo->array([ // x.shape = (2,2,2)
            [[1,2],
             [3,4]],
            [[5,6],
             [7,8]]
        ]);
        $this->assertEquals([2,2,2],$x->shape());
        $z = $la->einsum(' ...i->i...',$x);         // (a,b,c)->(c,a,b)
        $this->assertEquals([ 
            [[1,3],
             [5,7]],
            [[2,4],
             [6,8]]
        ],$z->toArray());
        $this->assertEquals($la->transpose($x,perm:[2,0,1])->toArray(),$z->toArray());

        // transpose complex array with blaceholder (left)
        $x = $la->range(start:0,limit:(2*3*4))->reshape([2,3,4]); // x.shape = (2,3,4)
        $this->assertEquals([2,3,4],$x->shape());
        $z = $la->einsum(' ...i->i...',$x); // (a,b,c)->(c,a,b)
        $this->assertEquals([4,2,3],$z->shape());
        $this->assertEquals($la->transpose($x,perm:[2,0,1])->toArray(),$z->toArray());

        // transpose with blaceholder (right)
        $x = $mo->array([ // x.shape = (2,2,2)
            [[1,2],
             [3,4]],
            [[5,6],
             [7,8]]
        ]);
        $this->assertEquals([2,2,2],$x->shape());
        $z = $la->einsum(' i...->...i',$x);     // (a,b,c)->(b,c,a)
        $this->assertEquals([ 
            [[1,5],
             [2,6]],
            [[3,7],
             [4,8]]
        ],$z->toArray());
        $this->assertEquals($la->transpose($x,perm:[1,2,0])->toArray(),$z->toArray());

        // transpose complex array with blaceholder (right)
        $x = $la->range(start:0,limit:(2*3*4))->reshape([2,3,4]); // x.shape = (2,3,4)
        $this->assertEquals([2,3,4],$x->shape());
        $z = $la->einsum(' i...->...i',$x);                 // (a,b,c)->(b,c,a)
        $this->assertEquals([3,4,2],$z->shape());
        $this->assertEquals($la->transpose($x,perm:[1,2,0])->toArray(),$z->toArray());

        // multi array placeholder (left)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]);         // x.shape = (2,2,2)
        $y = $mo->array([[[11,12],[13,14]],[[15,16],[17,18]]]); // y.shape = (2,2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,2,2],$y->shape());
        $z = $la->einsum(' ...i,...j-> ...ij',$x,$y);       // (abi,abj->abij)
        $this->assertEquals([
            [[[11,12],[22,24]],
             [[39,42],[52,56]]],
            [[[75,80],[90,96]],
             [[119,126],[136,144]]]
        ],$z->toArray());

        // multi array placeholder (left and right and left)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]);         // x.shape = (2,2,2)
        $y = $mo->array([[[11,12],[13,14]],[[15,16],[17,18]]]); // y.shape = (2,2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,2,2],$y->shape());
        $z = $la->einsum(' i...,...j-> ...ij',$x,$y);       // (iab,abj->abij)
        $this->assertEquals([
            [[[11,12],[55,60]],
             [[26,28],[78,84]]],
            [[[45,48],[105,112]],
             [[68,72],[136,144]]]
        ],$z->toArray());

        // multi array placeholder (left and right and right)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]);         // x.shape = (2,2,2)
        $y = $mo->array([[[11,12],[13,14]],[[15,16],[17,18]]]); // y.shape = (2,2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,2,2],$y->shape());
        $z = $la->einsum(' i...,...j-> ij...',$x,$y);   // (iab,abj->ijab)
        $this->assertEquals([
            [[[11,26],[45,68]],
             [[12,28],[48,72]]],
            [[[55,78],[105,136]],
             [[60,84],[112,144]]]
        ],$z->toArray());

        // broadcast mode with small array (left)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]); // x.shape = (2,2,2)
        $y = $mo->array([[9,10],[11,12]]);              // y.shape = (2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,2],$y->shape());
        $z = $la->einsum(' ...i,...i->...i ',$x,$y);    // (abi,bi->abi)
        $this->assertEquals([
            [[9,20],[33,48]],
            [[45,60],[77,96]],
        ],$z->toArray());

        // broadcast mode with small array (right)
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]); // x.shape = (2,2,2)
        $y = $mo->array([[9,10],[11,12]]);              // y.shape = (2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,2],$y->shape());
        $z = $la->einsum(' ...i,i...->...i ',$x,$y);    // (abi,ib->abi)
        $this->assertEquals([
            [[9,22],[30,48]],
            [[45,66],[70,96]],
        ],$z->toArray());

        // broadcast mode with Collapsed array
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]); // x.shape = (2,2,2)
        $y = $mo->array([[[9,10]],[[11,12]]]);          // y.shape = (2,1,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,1,2],$y->shape());
        $z = $la->einsum(' ...i,...i->...i ',$x,$y);    // (abi,a0i->abi)
        $this->assertEquals([
            [[9,20],[27,40]],
            [[55,72],[77,96]],
        ],$z->toArray());

        // placeholder only
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]); // x.shape = (2,2,2)
        $z = $la->einsum(' ...->... ',$x);    // (abc->abc)
        $this->assertEquals([[[1,2],[3,4]],[[5,6],[7,8]]],$z->toArray());

        // indices with placeholder to placeholder only
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]); // x.shape = (2,2,2)
        $z = $la->einsum(' ...i->... ',$x);    // (abc->abc)
        $this->assertEquals([[3,7],[11,15]],$z->toArray());

        // indices with placeholder and indices only
        $x = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]]); // x.shape = (2,2,2)
        $y = $mo->array([[9,10],[11,12]]);              // y.shape = (2,2)
        $this->assertEquals([2,2,2],$x->shape());
        $this->assertEquals([2,2],$y->shape());
        $z = $la->einsum(' ...i,ij->...j ',$x,$y);    // (abi,ij->abj)
        $this->assertEquals([
            [[31,34],[71,78]],
            [[111,122],[151,166]],
        ],$z->toArray());

    }

    public function testEinsum4p1Normal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // cross product 2D each dims
        $a = $mo->array([
            [0,1],
            [2,3],
            [4,5],
        ]);
        $b = $mo->array([
            [0,1,2,3,4],
            [5,6,7,8,9],
        ]);

        $a = $la->array($a);
        $b = $la->array($b);
        $this->assertEquals([3,2],$a->shape());
        $this->assertEquals([2,5],$b->shape());
        $c = $la->einsum4p1('ab,bc->ac',$a,$b,);
        $c = $la->toNDArray($c);
        $this->assertEquals([3,5],$c->shape());
        $this->assertEquals([
            [5,6,7,8,9],
            [15,20,25,30,35],
            [25,34,43,52,61]
        ],$c->toArray());

        // cross product 2D
        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1(' ik , kj -> ij ',$x,$y);
        $this->assertEquals([
            [19,22],
            [43,50]
        ],$z->toArray());

        // use cached equation
        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1(' ik , kj -> ij ',$x,$y);
        $this->assertEquals([
            [19,22],
            [43,50]
        ],$z->toArray());

        // transpose output
        $x = $mo->array([[1,2],[3,4]]);
        $y = $mo->array([[5,6],[7,8]]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1(' ik , kj -> ji ',$x,$y);
        $this->assertEquals([
            [19,43],
            [22,50]
        ],$z->toArray());

        // cross product 3D x 2D
        $x = $mo->array(range(0, 4*3*2-1))->reshape([4,3,2]);
        $y = $mo->array(range(0, 5*2-1))->reshape([5,2]);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1('abc,dc->abd',$x,$y);
        $this->assertEquals([4,3,5],$z->shape());
        $this->assertEquals([
            [[  1,   3,   5,   7,   9],
             [  3,  13,  23,  33,  43],
             [  5,  23,  41,  59,  77]],

            [[  7,  33,  59,  85, 111],
             [  9,  43,  77, 111, 145],
             [ 11,  53,  95, 137, 179]],

            [[ 13,  63, 113, 163, 213],
             [ 15,  73, 131, 189, 247],
             [ 17,  83, 149, 215, 281]],

            [[ 19,  93, 167, 241, 315],
             [ 21, 103, 185, 267, 349],
             [ 23, 113, 203, 293, 383]],
        ],$z->toArray());


        // multi head attention dot product 4D
        $shapeX = [2,2,2,2];
        $shapeY = [2,2,2,2];
        $x = $mo->array(range(0, (int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(0, (int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1('aecd,abcd->acbe',$x,$y);
        $this->assertEquals([2,2,2,2],$z->shape());
        $this->assertEquals([
          [[[  1,   5],
            [  5,  41]],
           [[ 13,  33],
            [ 33,  85]]],
          [[[145, 213],
            [213, 313]],
           [[221, 305],
            [305, 421]]],
        ],$z->toArray());

        // multi head attention combine 4D
        $shapeX = [2,2,2,2];
        $shapeY = [2,2,2,2];
        $x = $mo->array(range(0, (int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(0, (int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1('acbe,aecd->abcd',$x,$y);
        $this->assertEquals([2,2,2,2],$z->shape());
        $this->assertEquals([
          [[[  4,   5],
            [ 38,  47]],
           [[ 12,  17],
            [ 54,  67]]],
          [[[172, 189],
            [302, 327]],
           [[212, 233],
            [350, 379]]],
        ],$z->toArray());


        // share indicator 
        $shapeX = [2,2];
        $shapeY = [2,3];
        $x = $mo->array(range(1, 1+(int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(1, 1+(int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $z = $la->einsum4p1('aa,ab->ab',$x,$y);
        $this->assertEquals([2,3],$z->shape());
        //echo $mo->toString($z,indent:true)."\n";
        $this->assertEquals([
            [ 1,  2,  3],
            [16, 20, 24],
        ],$z->toArray());


        $shapeX = [2,2];
        $shapeY = [2,2];
        $x = $mo->array(range(1, 1+(int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(1, 1+(int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        //$this->expectException(InvalidArgumentException::class);
        //$this->expectExceptionMessage('rank of C must be less than 4D or equal.');
        $z = $la->einsum4p1('ab,ab->ab',$x,$y);
        $this->assertEquals([2,2],$z->shape());
        $this->assertEquals([
            [ 1,  4],
            [ 9, 16],
        ],$z->toArray());

    }

    public function testEinsum4p1OverRank()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // multi head attention dot product 5D to 6D
        $shapeX = [2,2,2,2,2];
        $shapeY = [2,2,2,2,2];
        $x = $mo->array(range(0, (int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(0, (int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('rank of C must be less than 4D or equal.');
        $z = $la->einsum4p1('afgde,abcde->adbcfg',$x,$y);
    }

    public function testEinsum4p1OverOverlayRank()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $shapeX = [2,2];
        $shapeY = [2,2];
        $x = $mo->array(range(1, 1+(int)array_product($shapeX)-1))->reshape($shapeX);
        $y = $mo->array(range(1, 1+(int)array_product($shapeY)-1))->reshape($shapeY);
        $x = $la->array($x);
        $y = $la->array($y);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage("rank of overlay dims must be less than 1D or equal. 'ab,ac->a' and 2 sum dims given.");
        $z = $la->einsum4p1('ab,ac->a',$x,$y);
        //echo $mo->toString($la->toNDArray($z),indent:true)."\n";
    }

    public function testMaskingNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //
        // Same shape (implicit batchDims:0, axis:0)
        //echo "==== Same shape (implicit batchDims:0, axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $la->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->masking($X,$A);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, 0,100],[0,-10, 0]]
        ,$A->toArray());

        //
        // broadcast to details
        //echo "==== broadcast to details\n";
        //
        // X:(2,3  )
        // A:(2,3,4)
        // outer:(2,3),bro:(4),inner:(),bro2:()
        // m=2*3,n=4,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=4
        $X = $la->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $la->array([
            [[1,11,111,1111],[2,12,122,1222],[-3,13,133,1333]],
            [[1,21,121,1211],[2,22,222,2222],[-3,23,233,2333]]
        ]);
        $la->masking($X,$A,batchDims:$X->ndim(),axis:$A->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,11,111,1111],[0, 0,  0,   0],[-3,13,133,1333]],
            [[0, 0,  0,   0],[2,22,222,2222],[0, 0,  0,   0]]
        ],$A->toArray());

        //
        // broadcast to details
        //echo "==== broadcast to details for implicit\n";
        //
        // X:(2,3  )
        // A:(2,3,4)
        // outer:(2,3),bro:(4),inner:(),bro2:()
        // m=2*3,n=4,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=4
        $X = $la->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $la->array([
            [[1,11,111,1111],[2,12,122,1222],[-3,13,133,1333]],
            [[1,21,121,1211],[2,22,222,2222],[-3,23,233,2333]]
        ]);
        $la->masking($X,$A,batchDims:$X->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,11,111,1111],[0, 0,  0,   0],[-3,13,133,1333]],
            [[0, 0,  0,   0],[2,22,222,2222],[0, 0,  0,   0]]
        ],$A->toArray());

        //
        // broadcast with gap
        //echo "==== broadcast with gap\n";
        //
        // X:(2,  3)
        // A:(2,4,3)
        // outer:(2),bro:(4),inner:(3),bro2:()
        // m=2,n=4,k=3,len=1
        $X = $la->array([
            [true,false,true],
            [false,true,false]
        ], dtype:NDArray::bool);
        $A = $la->array([
            [[1,11,111],[2,12,112],[-3,13,113],[-4,14,114]],
            [[1,21,211],[2,22,222],[-3,23,223],[-4,24,224]],
        ]);
        $la->masking($X,$A,batchDims:1,axis:2);
        $this->assertEquals([
            [true,false,true],
            [false,true,false]
        ],$X->toArray());
        $this->assertEquals([
            [[1, 0,111],[2, 0,112],[-3, 0,113],[-4, 0,114]],
            [[0,21,  0],[0,22,  0],[ 0,23,  0],[ 0,24,  0]],
        ],$A->toArray());

        //
        // broadcast to rows (implicit batchDims:0)
        //echo "==== broadcast to rows (implicit batchDims:0)\n";
        //
        // X:(  2,3)
        // A:(4,2,3)
        // outer:(),bro:(4),inner:(2,3),bro2:()
        // m=1,n=2,k=2*3,len=1
        $X = $la->array([[true,false,true],[false,true,false]],dtype:NDArray::bool);
        $A = $la->array([
            [[1,11,111],[2,12,112]],
            [[1,21,211],[2,22,222]],
            [[1,31,311],[2,32,322]],
            [[1,41,411],[2,42,422]],
        ]);
        $la->masking($X,$A,axis:1);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1, 0,111],[0,12,  0]],
            [[1, 0,211],[0,22,  0]],
            [[1, 0,311],[0,32,  0]],
            [[1, 0,411],[0,42,  0]],
        ],$A->toArray());

        //
        // broadcast to rows (implicit axis:batchDims)
        //echo "==== broadcast to rows (implicit axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(2),bro:(),inner:(3),bro2:()
        // m=2,n=1,k=3,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $la->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->masking($X,$A,batchDims:1);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, 0,100],[0,-10, 0]]
        ,$A->toArray());

        //
        // broadcast to rows (implicit batchDims:0, minus axis)
        //echo "==== broadcast to rows (implicit batchDims:0, minus axis)\n";
        //
        // X:(  2,3)
        // A:(4,2,3)
        // outer:(),bro:(4),inner:(2,3),bro2:()
        // m=1,n=4,k=2*3,len=1
        $X = $la->array([[true,false,true],[false,true,false]],dtype:NDArray::bool);
        $A = $la->array([
            [[1,11,111],[2,12,112]],
            [[1,21,211],[2,22,222]],
            [[1,31,311],[2,32,322]],
            [[1,41,411],[2,42,422]],
        ]);
        $la->masking($X,$A,axis:-$X->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1, 0,111],[0,12,  0]],
            [[1, 0,211],[0,22,  0]],
            [[1, 0,311],[0,32,  0]],
            [[1, 0,411],[0,42,  0]],
        ],$A->toArray());

        //
        // broadcast with gap and implicit len
        //echo "==== broadcast with gap implicit len\n";
        //
        // X:(2,  3  )
        // A:(2,4,3,2)
        // outer:(2),bro:(4),inner:(3),bro2:(2)
        // m=2,n=4,k=3,len=2
        $X = $la->array([
            [true,false,true],
            [false,true,false]
        ], dtype:NDArray::bool);
        $A = $la->array([
            [[[1,-1],[11,-11],[111,-111]],
             [[2,-2],[12,-12],[112,-112]],
             [[-3,3],[13,-13],[113,-113]],
             [[-4,4],[14,-14],[114,-114]]],
            [[[1,-1],[21,-21],[211,-211]],
             [[2,-2],[22,-22],[222,-222]],
             [[-3,3],[23,-23],[223,-223]],
             [[-4,4],[24,-24],[224,-224]]],
        ]);
        $la->masking($X,$A,batchDims:1,axis:2);
        $this->assertEquals([
            [true,false,true],
            [false,true,false]
        ],$X->toArray());
        $this->assertEquals([
            [[[1,-1],[ 0,  0],[111,-111]],
             [[2,-2],[ 0,  0],[112,-112]],
             [[-3,3],[ 0,  0],[113,-113]],
             [[-4,4],[ 0,  0],[114,-114]]],
            [[[0, 0],[21,-21],[  0,   0]],
             [[0, 0],[22,-22],[  0,   0]],
             [[0, 0],[23,-23],[  0,   0]],
             [[0, 0],[24,-24],[  0,   0]]],
        ],$A->toArray());

        //
        // broadcast to rows (implicit batchDims:0, implicit len)
        //echo "==== broadcast to rows (implicit batchDims:0, axis=1, implicit len)\n";
        //
        // X:(  2  )
        // A:(4,2,3)
        // outer:(),bro:(4),inner:(2),bro2:(3)
        // m=1,n=4,k=2,len=3
        $X = $la->array([true,false],dtype:NDArray::bool);
        $A = $la->array([
            [[1,11,111],[2,12,112]],
            [[1,21,211],[2,22,222]],
            [[1,31,311],[2,32,322]],
            [[1,41,411],[2,42,422]],
        ]);
        $la->masking($X,$A,axis:1);
        $this->assertEquals([true,false],$X->toArray());
        $this->assertEquals([
            [[1,11,111],[0, 0,  0]],
            [[1,21,211],[0, 0,  0]],
            [[1,31,311],[0, 0,  0]],
            [[1,41,411],[0, 0,  0]],
        ],$A->toArray());

        //
        // fill -9999
        //
        //echo "==== fill -9999\n";
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1  ==translate==> m=1,n=1,k=2*3
        $X = $la->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->masking($X,$A,fill:-9999);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, -9999,100],[-9999,-10, -9999]]
        ,$A->toArray());


    }

    public function testMaskingAddMode()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //
        // Same shape (implicit batchDims:0, axis:0)
        //echo "==== Same shape (implicit batchDims:0, axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $la->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->masking($X,$A, fill:-1000, mode:1); // 0:set mode,  1:add mode
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        //var_dump($A->toArray());
        $this->assertEquals([
            [1, -990,100],
            [-1001,-10, -1100]
        ],$A->toArray());
    }

    public function testMaskingBoolSetMode()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //
        // Same shape (implicit batchDims:0, axis:0)
        //echo "==== Same shape (implicit batchDims:0, axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $la->array([true, true, false,false,false,true ], dtype:NDArray::bool);
        $A = $la->array([true, false,true, false,true, true ], dtype:NDArray::bool);
        $la->masking($X,$A);
        $this->assertEquals(
            [true, true, false,false,false,true ]
        ,$X->toArray());
        $this->assertEquals(
            [true, false,false,false,false,true]
        ,$A->toArray());
    }

    public function testMaskingBoolAddMode()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //
        // Same shape (implicit batchDims:0, axis:0)
        //echo "==== Same shape (implicit batchDims:0, axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $la->array([true, true, false,false,false,true ], dtype:NDArray::bool);
        $A = $la->array([true, false,true, false,true, true ], dtype:NDArray::bool);
        $la->masking($X,$A,fill:true,mode:1); // mode=1:add
        $this->assertEquals(
            [true, true, false,false,false,true ]
        ,$X->toArray());
        $this->assertEquals(
            [true,false,true, true, true, true ]
        ,$A->toArray());
    }

}
