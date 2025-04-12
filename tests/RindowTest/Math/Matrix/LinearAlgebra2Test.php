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
