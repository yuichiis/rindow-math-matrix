<?php
namespace RindowTest\Math\Matrix\LinearAlgebraPHPModeTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpBlas;
use Rindow\Math\Matrix\Drivers\Service;
use ArrayObject;
use InvalidArgumentException;

if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\LinearAlgebraTest')) {
    require_once __DIR__.'/LinearAlgebraTest.php';
}
use RindowTest\Math\Matrix\LinearAlgebraTest\LinearAlgebraTest as ORGTest;

class LinearAlgebraPHPModeTest extends ORGTest
{
    static protected $speedtest = false;

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

    public function testBlasLevel2RejectsColMajor()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1.0,2.0],[3.0,4.0]]);
        $X = $la->array([1.0,1.0]);
        $blas = new PhpBlas();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid Order type');
        $blas->symv(
            BLAS::ColMajor,BLAS::Upper,2,1.0,
            $A->buffer(),$A->offset(),2,
            $X->buffer(),$X->offset(),1,
            0.0,
            $X->buffer(),$X->offset(),1);
    }


    //public function testSvdFull1()
    //{
    //    $this->markTestSkipped('Unsuppored function without openblas');
    //}

    //public function testSvdFull2()
    //{
    //    $this->markTestSkipped('Unsuppored function without openblas');
    //}

    //public function testSvdSmallVT()
    //{
    //    $this->markTestSkipped('Unsuppored function without openblas');
    //}
}
