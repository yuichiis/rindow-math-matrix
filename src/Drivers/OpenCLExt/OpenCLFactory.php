<?php
namespace Rindow\Math\Matrix\Drivers\OpenCLExt;

use Interop\Polite\Math\Matrix\LinearBuffer;
use Rindow\OpenCL\PlatformList;
use Rindow\OpenCL\Context;
use Rindow\OpenCL\CommandQueue;
use Rindow\OpenCL\DeviceList;
use Rindow\OpenCL\Program;
use Rindow\OpenCL\Kernel;
use Rindow\OpenCL\EventList;
use Rindow\Math\Matrix\OpenCLBuffer;

class OpenCLFactory
{
    protected string $extName = 'rindow_opencl';

    public function __construct()
    {
    }

    public function name() : string
    {
        return $this->extName();
    }

    public function isAvailable() : bool
    {
        return extension_loaded($this->extName());
    }

    public function extName() : string
    {
        return $this->extName;
    }

    public function version() : string
    {
        return phpversion($this->extName());
    }

    public function PlatformList() : PlatformList
    {
        return new PlatformList();
    }

    public function DeviceList(
        PlatformList $platforms,
        int $index=NULL,
        int $deviceType=NULL,
    ) : DeviceList
    {
        $index = $index ?? 0;
        $deviceType = $deviceType ?? 0;
        return new DeviceList($platforms,$index,$deviceType);
    }

    public function Context(
        DeviceList|int $arg
    ) : Context
    {
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
