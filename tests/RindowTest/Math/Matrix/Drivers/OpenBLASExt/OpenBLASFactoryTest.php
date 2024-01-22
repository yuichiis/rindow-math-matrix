<?php
namespace RindowTest\Math\Matrix\Drivers\OpenBLASExt\OpenBLASFactoryTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\OpenBLASExt\OpenBLASFactory;
use Rindow\Math\Matrix\Drivers\OpenBLASExt\OpenBlasBuffer;
use Rindow\OpenBLAS\Blas;
use Rindow\OpenBLAS\Lapack;
use Rindow\OpenBLAS\Math;
use Interop\Polite\Math\Matrix\NDArray;

/**
 * @requires extension rindow_openblas
 */
class Test extends TestCase
{
    public function newDriverFactory()
    {
        return new OpenBLASFactory();
    }

    public function testName()
    {
        $factory = $this->newDriverFactory();
        $this->assertEquals('rindow_openblas',$factory->name());
    }

    public function testIsAvailable()
    {
        $factory = $this->newDriverFactory();
        $this->assertTrue($factory->isAvailable());
    }

    public function testExtName()
    {
        $factory = $this->newDriverFactory();
        $this->assertEquals('rindow_openblas',$factory->extName());
    }

    public function testVersion()
    {
        $factory = $this->newDriverFactory();
        $this->assertTrue(is_string($factory->version()));
        //var_dump($factory->version());
    }

    public function testBlas()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->Blas();
        $this->assertInstanceOf(Blas::class,$driver);
    }

    public function testLapack()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->Lapack();
        $this->assertInstanceOf(Lapack::class,$driver);
    }

    public function testMath()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->Math();
        $this->assertInstanceOf(Math::class,$driver);
    }

    public function testBuffer()
    {
        $factory = $this->newDriverFactory();
        $size = 2;
        $dtype = NDArray::float32;
        $driver = $factory->Buffer($size,$dtype);
        $this->assertInstanceOf(OpenBlasBuffer::class,$driver);
    }

}
