<?php
namespace RindowTest\Math\Matrix\Drivers\OpenCLExt\OpenCLFactoryTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\OpenCLExt\OpenCLFactory;
use Rindow\Math\Matrix\Drivers\OpenCLExt\OpenCLBuffer;
use Rindow\OpenCL\PlatformList;
use Rindow\OpenCL\DeviceList;
use Rindow\OpenCL\Context;
use Rindow\OpenCL\EventList;
use Rindow\OpenCL\CommandQueue;
use Rindow\OpenCL\Program;
use Rindow\OpenCL\Kernel;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;

use Rindow\Math\Matrix\Drivers\OpenBLASExt\OpenBlasBuffer as HostBuffer;

use RuntimeException;
/**
 * @requires extension rindow_opencl
 */
class OpenCLFactoryTest extends TestCase
{
    static protected int $default_device_type = OpenCL::CL_DEVICE_TYPE_GPU;

    public function newDriverFactory()
    {
        $factory = new OpenCLFactory();
        return $factory;
    }

    public function newContextFromType($ocl)
    {
        try {
            $context = $ocl->Context(self::$default_device_type);
        } catch(RuntimeException $e) {
            if(strpos('clCreateContextFromType',$e->getMessage())===null) {
                throw $e;
            }
            self::$default_device_type = OpenCL::CL_DEVICE_TYPE_DEFAULT;
            $context = $ocl->Context(self::$default_device_type);
        }
        return $context;
    }

    public function testName()
    {
        $factory = $this->newDriverFactory();
        $this->assertEquals('rindow_opencl',$factory->name());
    }

    public function testIsAvailable()
    {
        $factory = $this->newDriverFactory();
        $this->assertTrue($factory->isAvailable());
    }

    public function testExtName()
    {
        $factory = $this->newDriverFactory();
        $this->assertEquals('rindow_opencl',$factory->extName());
    }

    public function testVersion()
    {
        $factory = $this->newDriverFactory();
        $this->assertTrue(is_string($factory->version()));
        //var_dump($factory->version());
    }

    public function testPlatformList()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->PlatformList();
        $this->assertInstanceOf(PlatformList::class,$driver);
    }

    public function testDeviceList()
    {
        $factory = $this->newDriverFactory();
        $platforms = $factory->PlatformList();
        $driver = $factory->DeviceList($platforms);
        $this->assertInstanceOf(DeviceList::class,$driver);
    }

    public function testContext()
    {
        $factory = $this->newDriverFactory();
        $platforms = $factory->PlatformList();
        $devices = $factory->DeviceList($platforms);
        $driver = $factory->Context($devices);
        $this->assertInstanceOf(Context::class,$driver);

        // Specify GPU directly
        $driver = $this->newContextFromType($factory);
        $this->assertInstanceOf(Context::class,$driver);
    }

    public function testEventList()
    {
        $factory = $this->newDriverFactory();
        $context = $this->newContextFromType($factory);
        $driver = $factory->EventList($context);
        $this->assertInstanceOf(EventList::class,$driver);
    }

    public function testCommandQueue()
    {
        $factory = $this->newDriverFactory();
        $context = $this->newContextFromType($factory);
        $driver = $factory->CommandQueue($context);
        $this->assertInstanceOf(CommandQueue::class,$driver);
    }

    public function testProgram()
    {
        $sources = [
            "typedef int number_int_t;\n".
            "__kernel void saxpy(const global float * x,\n".
            "                    __global float * y,\n".
            "                    const float a)\n".
            "{\n".
            "   uint gid = get_global_id(0);\n".
            "   y[gid] = a* x[gid] + y[gid];\n".
            "}\n"
        ];
    
        $factory = $this->newDriverFactory();
        $context = $this->newContextFromType($factory);
        $driver = $factory->Program(
            $context,
            $sources,
            0, // mode: source code
        );
        $this->assertInstanceOf(Program::class,$driver);
        $driver->compile();
    }

    public function testBuffer()
    {
        $size = 16;
        $dtype = NDArray::float32;
        $hostBuffer = new HostBuffer(
            $size,$dtype);
        foreach(range(0,$size-1) as $value) {
            $hostBuffer[$value] = $value;
        }
        
        $factory = $this->newDriverFactory();
        $context = $this->newContextFromType($factory);
        $driver = $factory->Buffer(
            $context,
            $size*32/8,
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBuffer,
            0, // hostbuffer offset
            $dtype,
        );
        $this->assertInstanceOf(OpenCLBuffer::class,$driver);
    }

    public function testKernel()
    {
        $sources = [
            "typedef int number_int_t;\n".
            "__kernel void saxpy_ext(const global float * x,\n".
            "                    __global float * y,\n".
            "                    const float a)\n".
            "{\n".
            "   uint gid = get_global_id(0);\n".
            "   y[gid] = a* x[gid] + y[gid];\n".
            "}\n"
        ];
    
        $factory = $this->newDriverFactory();
        $context = $this->newContextFromType($factory);

        $program = $factory->Program(
            $context,
            $sources,
            0, // mode: source code
        );
        $program->build();

        $driver = $factory->Kernel($program,"saxpy_ext");
        $this->assertInstanceOf(Kernel::class,$driver);
    }
}
