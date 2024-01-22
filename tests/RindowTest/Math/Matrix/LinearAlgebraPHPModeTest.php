<?php
namespace RindowTest\Math\Matrix\LinearAlgebraPHPModeTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibPhp;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\PhpBlas;
use Rindow\Math\Matrix\PhpLapack;
use Rindow\Math\Matrix\PhpMath;
use ArrayObject;
use InvalidArgumentException;

if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
    require_once __DIR__.'/LinearAlgebraTest.php';
}
use RindowTest\Math\Matrix\LinearAlgebraTest\Test as ORGTest;

class Test extends ORGTest
{
    public function setUp() : void
    {
        $this->service = new MatlibPhp();
        if($this->service->serviceLevel()!=Service::LV_BASIC) {
            throw new \Exception("the service is invalid.");
        }
    }

    public function testTrsmNormal()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }


    public function testSvdFull1()
    {
        $this->markTestSkipped('Unsuppored function without openblas');
    }

    public function testSvdFull2()
    {
        $this->markTestSkipped('Unsuppored function without openblas');
    }

    public function testSvdSmallVT()
    {
        $this->markTestSkipped('Unsuppored function without openblas');
    }
}
