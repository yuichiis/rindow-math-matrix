<?php
namespace RindowTest\Math\Matrix\RandomTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use ArrayObject;
use InvalidArgumentException;

class RandomTest extends TestCase
{
    public function newMatrixOperator()
    {
        $mo = new MatrixOperator();
        return $mo;
    }

    public function testRand()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->random()->rand(10000);
        $this->assertGreaterThanOrEqual(0,$mo->min($x));
        $this->assertLessThanOrEqual(1,$mo->max($x));
    }

    public function testRandn()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->random()->randn(10000);
        $this->assertTrue(true);
    }

    public function testRandomInt()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->random()->randomInt(100);
        $this->assertTrue(true);
    }

    public function testChoiceNormal()
    {
        $mo = $this->newMatrixOperator();

        $choice = $mo->random()->choice($total=10, $sampling=4,$replace=false);
        $this->assertEquals([4],$choice->shape());

        $count = $mo->zeros([10]);
        $mo->update($count,'+=',1,$choice);
        $this->assertEquals(6,$mo->sum($mo->op($count,'==',0)));
        $this->assertEquals(4,$mo->sum($mo->op($count,'==',1)));

        $choice = $mo->random()->choice($total=10, $sampling=4,$replace=true);
        $this->assertEquals([4],$choice->shape());

        $choice = $mo->random()->choice($total=4, $sampling=10,$replace=true);
        $this->assertEquals([10],$choice->shape());
    }

    public function testChoiceFromSource()
    {
        $mo = $this->newMatrixOperator();
        $source = $mo->array([10,11,12,13,14],dtype:NDArray::int32);

        $choice = $mo->random()->choice($source,$sampling=1);
        $this->assertTrue(is_int($choice));

        $choice = $mo->random()->choice($source,$sampling=2);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$choice);
        $this->assertEquals([2],$choice->shape());

        $source = $mo->array([10]);

        $choice = $mo->random()->choice($source,$sampling=1);
        $this->assertEquals(10,$choice);
        $choice = $mo->random()->choice($source,$sampling=2);
        $this->assertEquals([10,10],$choice->toArray());
        $choice = $mo->random()->choice($source,$sampling=2,$replace=true);
        $this->assertEquals([10,10],$choice->toArray());
    }

    public function testChoiceSmallBaseTotal()
    {
        $mo = $this->newMatrixOperator();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The total number is smaller than the number of samples');
        $choice = $mo->random()->choice($total=4, $sampling=10,$replace=false);
    }

    public function testChoiceNoReplace()
    {
        $mo = $this->newMatrixOperator();
        $size = 10000;
        $choice = $mo->random()->choice($total=$size, $sampling=$size,$replace=false);
        $seen = [];
        for($i=0;$i<$size;$i++) {
            $seen[$choice[$i]] = 1;
        }
        $this->assertCount($size,$seen);

        $size = 10000;
        $choice = $mo->random()->choice($total=$size*2, $sampling=$size,$replace=false);
        $seen = [];
        for($i=0;$i<$size;$i++) {
            $seen[$choice[$i]] = 1;
        }
        $this->assertCount($size,$seen);
        $count=0;
        foreach($seen as $k=>$v) {
            if($k>$size) {
                if($k>=$size*2) {
                    $this->assertTrue(false);
                }
                $count++;
            }
        }
        $this->assertGreaterThanOrEqual(3000,$count);
    }
}
