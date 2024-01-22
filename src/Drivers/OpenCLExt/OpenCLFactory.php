<?php
namespace Rindow\Math\Matrix\Drivers\OpenCLExt;

use Interop\Polite\Math\Matrix\LinearBuffer;
use Interop\Polite\Math\Matrix\Buffer;
use Rindow\OpenCL\PlatformList;
use Rindow\OpenCL\Context;
use Rindow\OpenCL\CommandQueue;
use Rindow\OpenCL\DeviceList;
use Rindow\OpenCL\Program;
use Rindow\OpenCL\Kernel;
use Rindow\OpenCL\EventList;
use Rindow\Math\Matrix\Drivers\AbstractDriver;

class OpenCLFactory extends AbstractDriver
{
    protected string $LOWEST_VERSION = '0.2.0';
    protected string $OVER_VERSION   = '0.3.0';

    protected string $extName = 'rindow_opencl';

    public function PlatformList() : PlatformList
    {
        $this->assertVersion();
        return new PlatformList();
    }

    public function DeviceList(
        PlatformList $platforms,
        int $index=NULL,
        int $deviceType=NULL,
    ) : DeviceList
    {
        $this->assertVersion();
        $index = $index ?? 0;
        $deviceType = $deviceType ?? 0;
        return new DeviceList($platforms,$index,$deviceType);
    }

    public function Context(
        DeviceList|int $arg
    ) : Context
    {
        $this->assertVersion();
        return new Context($arg);
    }

    public function EventList(
        Context $context=null
    ) : EventList
    {
        return new EventList($context);
    }

    public function CommandQueue(
        Context $context,
        long $deviceId=null,
        long $properties=null,
    ) : CommandQueue
    {
        $deviceId = $deviceId ?? 0;
        $properties = $properties ?? 0;
        return new CommandQueue($context, $deviceId, $properties);
    }

    public function Program(
        Context $context,
        string|array $source,   // string or list of something
        int $mode=null,         // mode  0:source codes, 1:binary, 2:built-in kernel, 3:linker
        DeviceList $deviceList=null,
        string $options=null,
        ) : Program
    {
        $mode = $mode ?? 0;
        return new Program($context, $source, $mode, $deviceList, $options);
    }

    public function Buffer(
        Context $context,
        int $size,
        int $flags=null,
        LinearBuffer $hostBuffer=null,
        int $hostOffset=null,
        int $dtype=null,
        ) : Buffer
    {
        return new OpenCLBuffer($context, $size, $flags, $hostBuffer, $hostOffset, $dtype);
    }

    public function Kernel
    (
        Program $program,
        string $kernelName,
        ) : Kernel
    {
        return new Kernel($program, $kernelName);
    }
}
