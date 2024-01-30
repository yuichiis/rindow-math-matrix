<?php
namespace RindowTest\Math\Matrix\Drivers\PhpBLAS\PhpBlasPHPModeTest;

if(!class_exists('RindowTest\Math\Matrix\Drivers\PhpBLAS\PhpBlasTest\Test')) {
    include __DIR__.'/PhpBlasTest.php';
}
use RindowTest\Math\Matrix\Drivers\PhpBLAS\PhpBlasTest\PhpBlasTest as ORGTest;
use Rindow\Math\Matrix\PhpBlas;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\MatrixOperator;

class PhpBlasPHPModeTest extends ORGTest
{
    public function getBlas($mo)
    {
        $blas = $mo->service()->blas(Service::LV_BASIC);
        //$blas = $mo->blas();
        return $blas;
    }

    public function testGetConfig()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertStringStartsWith('PhpBlas',$blas->getConfig());
    }

    public function testGetNumThreads()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertEquals(1,$blas->getNumThreads());
    }

    public function testGetNumProcs()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertEquals(1,$blas->getNumProcs());
    }

    public function testGetCorename()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertTrue(is_string($blas->getCorename()));
    }
}
