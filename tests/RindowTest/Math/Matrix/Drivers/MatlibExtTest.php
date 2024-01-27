<?php
namespace RindowTest\Math\Matrix\Drivers\MatlibExtTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\MatlibExt;

use Rindow\Math\Matrix\Drivers\OpenBLASExt\OpenBLASFactory;
use Rindow\Math\Matrix\Drivers\OpenBLASExt\OpenBlasBuffer;
use Rindow\OpenBLAS\Blas;
use Rindow\OpenBLAS\Lapack;
use Rindow\OpenBLAS\Math;

use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBLASFactory;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBuffer;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBlas;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpLapack;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpMath;

use Rindow\Math\Matrix\Drivers\Service;


use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;

use InvalidArgumentException;

/**
 * @requires extension rindow_openblas
 */
class Test extends TestCase
{
    public function newService()
    {
        return new MatlibExt();
    }

    public function testName()
    {
        $service = $this->newService();
        $this->assertEquals('matlib_ext',$service->name());
    }

    public function testServiceLevel()
    {
        $service = $this->newService();
        if(extension_loaded('rindow_openblas')&&
            extension_loaded('rindow_opencl')&&
            extension_loaded('rindow_clblast')) {
            $this->assertEquals(Service::LV_ACCELERATED,$service->serviceLevel());
            //var_dump("LV_ACCELERATED");
        } elseif(extension_loaded('rindow_openblas')&&
            (!extension_loaded('rindow_opencl')||
            !extension_loaded('rindow_clblast'))) {
            $this->assertEquals(Service::LV_ADVANCED,$service->serviceLevel());
            //var_dump("LV_ADVANCED");
        } elseif(!extension_loaded('rindow_openblas')) {
            $this->assertEquals(Service::LV_BASIC,$service->serviceLevel());
            //var_dump("LV_BASIC");
        }
    }

    public function testBlas()
    {
        $service = $this->newService();
        if(extension_loaded('rindow_openblas')) {
            $this->assertInstanceOf(Blas::class,$service->blas());
        } else {
            $this->assertInstanceOf(PhpBlas::class,$service->blas());
        }
        $this->assertInstanceOf(PhpBlas::class,$service->blas(Service::LV_BASIC));
    }

    public function testLapack()
    {
        $service = $this->newService();
        if(extension_loaded('rindow_openblas')) {
            $this->assertInstanceOf(Lapack::class,$service->lapack());
        } else {
            $this->assertInstanceOf(PhpLapack::class,$service->lapack());
        }
        $this->assertInstanceOf(PhpLapack::class,$service->lapack(Service::LV_BASIC));
    }

    public function testMath()
    {
        $service = $this->newService();
        if(extension_loaded('rindow_openblas')) {
            $this->assertInstanceOf(Math::class,$service->math());
        } else {
            $this->assertInstanceOf(PhpMath::class,$service->math());
        }
        $this->assertInstanceOf(PhpMath::class,$service->math(Service::LV_BASIC));
    }

    public function testBuffer()
    {
        $service = $this->newService();
        $size = 2;
        $dtype = NDArray::float32;
        if(extension_loaded('rindow_openblas')) {
            $this->assertInstanceOf(OpenBlasBuffer::class,$service->buffer()->Buffer($size,$dtype));
        } else {
            $this->assertInstanceOf(PhpBuffer::class,$service->buffer()->Buffer($size,$dtype));
        }
        $this->assertInstanceOf(PhpBuffer::class,$service->buffer(Service::LV_BASIC)->Buffer($size,$dtype));
    }

    public function testCreateQueuebyDeviceType()
    {
        $service = $this->newService();
        if($service->serviceLevel()<Service::LV_ACCELERATED) {
            $this->markTestSkipped("The service is not Accelerated.");
            return;
        }
        try {
            $queue = $service->createQueue(['deviceType'=>OpenCL::CL_DEVICE_TYPE_GPU]);
        } catch(InvalidArgumentException $e) {
            $queue = $service->createQueue(['deviceType'=>OpenCL::CL_DEVICE_TYPE_CPU]);
        }
        $this->assertInstanceOf(\Rindow\OpenCL\CommandQueue::class,$queue);
        $this->assertInstanceOf(\Rindow\CLBlast\Blas::class,$service->blasCL($queue));
        $this->assertInstanceOf(\Rindow\Math\Matrix\Drivers\MatlibCL\OpenCLMath::class,$service->mathCL($queue));
        $this->assertInstanceOf(\Rindow\CLBlast\Math::class,$service->mathCLBlast($queue));
    }

    public function testCreateQueuebyDeviceId()
    {
        $service = $this->newService();
        if($service->serviceLevel()<Service::LV_ACCELERATED) {
            $this->markTestSkipped("The service is not Accelerated.");
            return;
        }
        $queue = $service->createQueue(['device'=>"0,0"]);
        $this->assertInstanceOf(\Rindow\OpenCL\CommandQueue::class,$queue);
        $this->assertInstanceOf(\Rindow\CLBlast\Blas::class,$service->blasCL($queue));
        $this->assertInstanceOf(\Rindow\Math\Matrix\Drivers\MatlibCL\OpenCLMath::class,$service->mathCL($queue));
        $this->assertInstanceOf(\Rindow\CLBlast\Math::class,$service->mathCLBlast($queue));
    }
}
