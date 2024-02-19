<?php
namespace RindowTest\Math\Matrix\NDArrayPhpTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\NDArrayPhp;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBuffer;
use Rindow\Math\Matrix\Drivers\Selector;
use Rindow\Math\Matrix\Drivers\Service;
use function Rindow\Math\Matrix\R;

use ArrayObject;
use OutOfRangeException;
use InvalidArgumentException;
use LogicException;

class NDArrayPhpTest extends TestCase
{
    protected $service;

    public function setUp() : void
    {
        $selector = new Selector();
        $this->service = $selector->select();
    }

    public function bufferToArray($buffer)
    {
        $size = count($buffer);
        $array = [];
        for($i=0;$i<$size;$i++) {
            $array[] = $buffer[$i];
        }
        return $array;
    }
    public function testCreateFromArray()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->assertEquals([2,3],$nd->shape());
        $this->assertEquals(6,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([1,2,3,4,5,6],$this->bufferToArray($buffer));
        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());

        $array = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]];
        $nd = new NDArrayPhp($array,NDArray::float64,service:$this->service);
        $this->assertEquals([2,3,2],$nd->shape());
        $this->assertEquals(12,$nd->size());
        $this->assertEquals(NDArray::float64,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(12,count($buffer));
        $this->assertEquals([1,2,3,4,5,6,7,8,9,10,11,12],$this->bufferToArray($buffer));
        $this->assertEquals([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]],$nd->toArray());

        $array = [1,2,3];
        $nd = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([3],$nd->shape());
        $this->assertEquals(3,$nd->size());
        $this->assertEquals(NDArray::int32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(3,count($buffer));
        $this->assertEquals([1,2,3],$this->bufferToArray($buffer));
        $this->assertEquals([1,2,3],$nd->toArray());
    }

    public function testCreateFromArrayObject()
    {
        $array = new ArrayObject([new ArrayObject([1,2,3]),new ArrayObject([4,5,6])]);
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->assertEquals([2,3],$nd->shape());
        $this->assertEquals(6,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([1,2,3,4,5,6],$this->bufferToArray($buffer));
        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());
    }

    public function testCreateFromNumber()
    {
        $array = 123;
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->assertEquals([],$nd->shape());
        $this->assertEquals(1,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(1,count($buffer));
        $this->assertEquals([123],$this->bufferToArray($buffer));
        $this->assertEquals(123,$nd->toArray());
    }

    public function testCreateNoInitialize()
    {
        $nd = new NDArrayPhp(null,NDArray::float32,[2,3],service:$this->service);
        $this->assertEquals([2,3],$nd->shape());
        $this->assertEquals(6,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([null,null,null,null,null,null],$this->bufferToArray($buffer));
        $this->assertEquals([[null,null,null],[null,null,null]],$nd->toArray());
    }

    public function testCreateFromBuffer()
    {
        $array = PhpBuffer::fromArrayWithDtype([1,2,3,4,5,6],NDArray::float32);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],3,service:$this->service);
        $this->assertEquals([1,3],$nd->shape());
        $this->assertEquals(3,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(3,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([1,2,3,4,5,6],$this->bufferToArray($buffer));
        $this->assertEquals([[4,5,6]],$nd->toArray());
    }

    public function testCreateFromIllegalArray()
    {
        $array = [[1,2,3],[4,5]];
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The shape of the dimension is broken');
        $nd = new NDArrayPhp($array,service:$this->service);
    }

    public function testCreateFromNumberAndIllegalShape()
    {
        $array = 123;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid dimension size');
        $nd = new NDArrayPhp($array,NDArray::float32,[2,3],service:$this->service);
    }

    public function testCreateNullAndNoShape()
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid type of array');
        $nd = new NDArrayPhp(null,NDArray::float32,service:$this->service);
    }

    public function testCreateFromBufferAndIllegalOffset()
    {
        $array = PhpBuffer::fromArrayWithDtype([1,2,3,4,5,6],NDArray::float32);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid dimension size');
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],4,service:$this->service);
    }

    public function testOffsetExists()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->assertTrue($nd->offsetExists(0));
        $this->assertTrue($nd->offsetExists(1));
        $this->assertFalse($nd->offsetExists(2));
        $this->assertFalse($nd->offsetExists(-1));
        $this->assertTrue(isset($nd[1][2]));
        $this->assertFalse(isset($nd[1][3]));
        $this->assertFalse(isset($nd[2][1]));
    }

    public function testOffsetExistsWithInvalidNumber()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->expectException(OutOfRangeException::class);
        $this->expectExceptionMessage('Dimension must be integer');
        $b = $nd->offsetExists(0.5);
    }

    public function testOffsetGet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->assertEquals(1,$nd[0][0]);
        $this->assertEquals(6,$nd[1][2]);
        $this->assertEquals([1,2,3],$nd[0]->toArray());
        $this->assertEquals([4,5,6],$nd[1]->toArray());
        $this->assertEquals(0,$nd[0]->offset());
        $this->assertEquals(3,$nd[1]->offset());
        $this->assertEquals(
            spl_object_hash($nd[0]->buffer()),
            spl_object_hash($nd[1]->buffer()));

        $array = PhpBuffer::fromArrayWithDtype([1,2,3,4,5,6],NDArray::float32);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],3,service:$this->service);
        $this->assertEquals(6,$nd[0][2]);
        $this->assertEquals([4,5,6],$nd[0]->toArray());

        $array = [[[1,2],[3,4]],[[5,6],[7,8]]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->assertEquals(8,$nd[1][1][1]);
        $this->assertEquals([7,8],$nd[1][1]->toArray());
        $this->assertEquals([[5,6],[7,8]],$nd[1]->toArray());

    }

    public function testOffsetSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $nd[0][0] = 10;
        $this->assertEquals([[10,2,3],[4,5,6]],$nd->toArray());
        $nd[1][2] = 60;
        $this->assertEquals([[10,2,3],[4,5,60]],$nd->toArray());
        $nd[1] = new NDArrayPhp([400,500,600],service:$this->service);
        $this->assertEquals([[10,2,3],[400,500,600]],$nd->toArray());

        $array = PhpBuffer::fromArrayWithDtype([1,2,3,4,5,6],NDArray::float32);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],3,service:$this->service);
        $nd[0][0] = 40;
        $this->assertEquals([[40,5,6]],$nd->toArray());
        $nd[0] = new NDArrayPhp([400,500,600],service:$this->service);
        $this->assertEquals([[400,500,600]],$nd->toArray());
    }

    public function testOutOfRangeWithOffgetSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->expectException(OutOfRangeException::class);
        $this->expectExceptionMessage('Index is out of range');
        $a = $nd[2];
    }

    public function testOutOfRangeWithOffsetSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->expectException(OutOfRangeException::class);
        $this->expectExceptionMessage('Index is out of range');
        $nd[3] = new NDArrayPhp([400,500,600],service:$this->service);
    }

    public function testOffunSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $this->expectException(LogicException::class);
        $this->expectExceptionMessage('Unsuppored Operation');
        unset($nd[1]);
    }

    public function testReshape()
    {
        $array = [[[0,0,0],[0,0,0]],[[1,2,3],[4,5,6]],[[0,0,0],[0,0,0]]];
        $nd = new NDArrayPhp($array,service:$this->service);
        $nd = $nd[1];
        $r = $nd->reshape([6]);
        $this->assertEquals([1,2,3,4,5,6],$r->toArray());
        $this->assertNotEquals(spl_object_hash($r),spl_object_hash($nd));
        $this->assertEquals([6],$r->shape());
        $this->assertEquals(spl_object_hash($r->buffer()),spl_object_hash($nd->buffer()));
    }

    public function testSerializeDefault()
    {
        $array = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([2,2,3],$a->shape());
        $this->assertEquals(NDArray::int32,$a->dtype());

        // latest style
        $str = $a->serialize();
        if($this->service->serviceLevel()>=Service::LV_ADVANCED) {
            $this->assertTrue(strpos($str,'inear-array')===false);
        }
        $b = new NDArrayPhp(service:$this->service);
        $b->unserialize($str);
        $this->assertEquals([2,2,3],$b->shape());
        $this->assertEquals(NDArray::int32,$b->dtype());
        $this->assertEquals($array,$b->toArray());

        // old style
        NDArrayPhp::$unserializeWarning = 0;
        try {
            $str = serialize($a);
            if($this->service->serviceLevel()>=Service::LV_ADVANCED) {
                $this->assertTrue(strpos($str,'inear-array')===false);
            }
            $b = unserialize($str);
            $this->assertEquals([2,2,3],$b->shape());
            $this->assertEquals(NDArray::int32,$b->dtype());
            $this->assertEquals($array,$b->toArray());
        } finally {
            NDArrayPhp::$unserializeWarning = 2;
        }

        // Unmatch methods
        NDArrayPhp::$unserializeWarning = 0;
        try {
            $str = serialize($a);
            if($this->service->serviceLevel()>=Service::LV_ADVANCED) {
                $this->assertTrue(strpos($str,'inear-array')===false);
            }
            $b = new NDArrayPhp(service:$this->service);
            $b->unserialize($str);
            $this->assertEquals([2,2,3],$b->shape());
            $this->assertEquals(NDArray::int32,$b->dtype());
            $this->assertEquals($array,$b->toArray());
        } finally {
            NDArrayPhp::$unserializeWarning = 2;
        }
    }

    public function testSerializePortable()
    {
        $array = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([2,2,3],$a->shape());
        $this->assertEquals(NDArray::int32,$a->dtype());
        $this->assertInstanceof(BufferInterface::class,$a->buffer());
        $this->assertFalse($a->getPortableSerializeMode());
        $a->setPortableSerializeMode(true);
        $this->assertTrue($a->getPortableSerializeMode());

        // latest style
        $str = $a->serialize();
        $this->assertTrue(strpos($str,'inear-array')!==null);

        $b = new NDArrayPhp(service:$this->service);
        $b->unserialize($str);
        $this->assertEquals([2,2,3],$b->shape());
        $this->assertEquals(NDArray::int32,$b->dtype());
        $this->assertEquals($array,$b->toArray());
        $this->assertInstanceof(BufferInterface::class,$b->buffer());
        $this->assertFalse($b->getPortableSerializeMode());

        // old style
        $str = serialize($a);
        $this->assertTrue(strpos($str,'inear-array')!==null);

        NDArrayPhp::$unserializeWarning = 0;
        try {
            $b = unserialize($str);
            $this->assertEquals([2,2,3],$b->shape());
            $this->assertEquals(NDArray::int32,$b->dtype());
            $this->assertEquals($array,$b->toArray());
            $this->assertInstanceof(BufferInterface::class,$b->buffer());
            $this->assertFalse($b->getPortableSerializeMode());
        } finally {
            NDArrayPhp::$unserializeWarning = 2;
        }
}

    public function testOffsetGetForRange()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([3,4,5],$a[R(2,5)]->toArray());
        $this->assertEquals(
            spl_object_hash($a->buffer()),
            spl_object_hash($a[R(2,5)]->buffer()));

        $this->assertEquals([3],$a[R(2,3)]->toArray());
        $this->assertEquals(
            spl_object_hash($a->buffer()),
            spl_object_hash($a[R(2,3)]->buffer()));

        $this->assertEquals([1,2,3,4,5,6,7,8,9,10],$a[R(0,10)]->toArray());
        $this->assertEquals(
            spl_object_hash($a->buffer()),
            spl_object_hash($a[R(0,10)]->buffer()));

        $array = [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([[[1,2],[3,4]],[[5,6],[7,8]]],$a[R(0,2)]->toArray());
        $this->assertEquals([[[5,6],[7,8]],[[9,10],[11,12]]],$a[R(1,3)]->toArray());
    }

    public function testOffsetExistsForRange()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertTrue($a->offsetExists(R(2,5)));
        $this->assertTrue($a->offsetExists(R(0,5)));
        $this->assertTrue($a->offsetExists(R(4,10)));
        $this->assertTrue($a->offsetExists(R(0,10)));
        $this->assertFalse($a->offsetExists(R(-1,10)));
        $this->assertFalse($a->offsetExists(R(0,11)));
    }

    public function testOffsetExistsForIllegalRange()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->expectException(OutOfRangeException::class);
        $this->expectExceptionMessage('Illegal range specification');
        $b = $a->offsetExists(R(1,1));
    }

    public function testCount()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertCount(10,$a);

        $array = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertCount(10,$a);
    }

    public function testIterator()
    {
        $array = [1,2,3,4,5];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([5],$a->shape());
        $r=[];
        foreach ($a as $key => $value) {
            $r[] = [$key,$value];
        }
        $this->assertEquals([
            [0,1],[1,2],[2,3],[3,4],[4,5]
        ],$r);

        $array = [[1],[2]];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([2,1],$a->shape());
        $r=[];
        foreach ($a as $key => $value) {
            $r[] = ['key'=>$key,'value'=>$value];
        }
        $this->assertEquals(0,$r[0]['key']);
        $this->assertInstanceof(NDArray::class,$r[0]['value']);
        $this->assertEquals([1],$r[0]['value']->toArray());

        $this->assertEquals(1,$r[1]['key']);
        $this->assertInstanceof(NDArray::class,$r[1]['value']);
        $this->assertEquals([2],$r[1]['value']->toArray());
    }

    public function testNestIterator()
    {
        $array = [[1,2],[3,4]];
        $a = new NDArrayPhp($array,NDArray::int32,service:$this->service);
        $this->assertEquals([2,2],$a->shape());
        $r=[];
        foreach ($a as $akey => $b) {
            foreach ($b as $bkey => $value) {
                $r[]='a['.$akey.']['.$bkey.']='.$value;
            }
        }
        $this->assertEquals([
            'a[0][0]=1',
            'a[0][1]=2',
            'a[1][0]=3',
            'a[1][1]=4',
        ],$r);
    }

    public function testClone()
    {
        $array = new NDArrayPhp([1,2],NDArray::int32,service:$this->service);
        $array2 = clone $array;
        $array[0] = 0;
        $array[1] = 0;
        $this->assertEquals(NDArray::int32,$array2->dtype());
        $buffer = $array2->buffer();
        //if(!($buffer instanceof PhpBuffer)) {
            $this->assertEquals(NDArray::int32,$buffer->dtype());
        //}
        $this->assertEquals(2,count($buffer));
        $this->assertEquals(1,$array2[0]);
        $this->assertEquals(2,$array2[1]);
    }
}
