<?php
namespace RindowTest\Math\Matrix\Drivers\MatlibFFITest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\MatlibFFI;

use Rindow\Math\Matrix\Drivers\OpenBLASExt\OpenBLASFactory;
use Rindow\Math\Buffer\FFI\Buffer;
use Rindow\OpenBLAS\FFI\Blas;
use Rindow\OpenBLAS\FFI\Lapack;
use Rindow\Matlib\FFI\Matlib as Math;

use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBLASFactory;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBuffer;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpBlas;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpLapack;
use Rindow\Math\Matrix\Drivers\PhpBLAS\PhpMath;

use Rindow\Math\Matrix\Drivers\Service;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;

use FFI\Env\Runtime as FFIEnvRuntime;
use FFI\Env\Status as FFIEnvStatus;
use FFI\Location\Locator as FFIEnvLocator;

/**
 * @requires extension ffi
 */
class Test extends TestCase
{
    public function newService()
    {
        return new MatlibFFI();
    }

    public function isAvailable(array $libs) : bool
    {
        $isAvailable = FFIEnvRuntime::isAvailable();
        if(!$isAvailable) {
            return false;
        }
        $pathname = FFIEnvLocator::resolve(...$libs);
        return $pathname!==null;
    }

    public function testName()
    {
        $service = $this->newService();
        $this->assertEquals('matlib_ffi',$service->name());
    }

    public function testServiceLevel()
    {
        $service = $this->newService();
        if($this->isAvailable(['libopenblas.dll','libopenblas.so'])&&
            $this->isAvailable(['matlib.dll','librindowmatlib.so'])&&
            $this->isAvailable(['OpenCL.dll','libopencl.so'])&&
            $this->isAvailable(['clblast.dll','libclblast.so'])) {
            $this->assertEquals(Service::LV_ACCELERATED,$service->serviceLevel());
        } elseif($this->isAvailable(['libopenblas.dll','libopenblas.so'])&&
            $this->isAvailable(['matlib.dll','librindowmatlib.so']) &&
            (!$this->isAvailable(['OpenCL.dll','libopencl.so'])||
            !$this->isAvailable(['clblast.dll','libclblast.so']))) {
            $this->assertEquals(Service::LV_ADVANCED,$service->serviceLevel());
        } elseif(!$this->isAvailable(['libopenblas.dll','libopenblas.so'])||
                !$this->isAvailable(['matlib.dll','librindowmatlib.so'])) {
            $this->assertEquals(Service::LV_BASIC,$service->serviceLevel());
        }
    }

    public function testBlas()
    {
        $service = $this->newService();
        if($service->serviceLevel()>=Service::LV_ADVANCED) {
            $this->assertInstanceOf(Blas::class,$service->blas());
        } else {
            $this->assertInstanceOf(PhpBlas::class,$service->blas());
        }
        $this->assertInstanceOf(PhpBlas::class,$service->blas(Service::LV_BASIC));
    }

    public function testLapack()
    {
        $service = $this->newService();
        if($service->serviceLevel()>=Service::LV_ADVANCED) {
            $this->assertInstanceOf(Lapack::class,$service->lapack());
        } else {
            $this->assertInstanceOf(PhpLapack::class,$service->lapack());
        }
        $this->assertInstanceOf(PhpLapack::class,$service->lapack(Service::LV_BASIC));
    }

    public function testMath()
    {
        $service = $this->newService();
        if($service->serviceLevel()>=Service::LV_ADVANCED) {
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
        if($service->serviceLevel()>=Service::LV_ADVANCED) {
            $this->assertInstanceOf(Buffer::class,$service->buffer()->Buffer($size,$dtype));
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
        $queue = $service->createQueue(['deviceType'=>OpenCL::CL_DEVICE_TYPE_GPU]);
        $this->assertInstanceOf(\Rindow\OpenCL\FFI\CommandQueue::class,$queue);
        $this->assertInstanceOf(\Rindow\CLBlast\FFI\Blas::class,$service->blasCL($queue));
        $this->assertInstanceOf(\Rindow\Math\Matrix\Drivers\MatlibCL\OpenCLMath::class,$service->mathCL($queue));
        $this->assertInstanceOf(\Rindow\CLBlast\FFI\Math::class,$service->mathCLBlast($queue));
    }

    public function testCreateQueuebyDeviceId()
    {
        $service = $this->newService();
        if($service->serviceLevel()<Service::LV_ACCELERATED) {
            $this->markTestSkipped("The service is not Accelerated.");
            return;
        }
        $queue = $service->createQueue(['device'=>"0,1"]);
        $this->assertInstanceOf(\Rindow\OpenCL\FFI\CommandQueue::class,$queue);
        $this->assertInstanceOf(\Rindow\CLBlast\FFI\Blas::class,$service->blasCL($queue));
        $this->assertInstanceOf(\Rindow\Math\Matrix\Drivers\MatlibCL\OpenCLMath::class,$service->mathCL($queue));
        $this->assertInstanceOf(\Rindow\CLBlast\FFI\Math::class,$service->mathCLBlast($queue));
    }
}
