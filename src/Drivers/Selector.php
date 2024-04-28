<?php
namespace Rindow\Math\Matrix\Drivers;

use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use LogicException;
use RuntimeException;

class Selector
{
    /** @var array<string> $catalog */
    protected array $catalog;
    protected ?Service $recommended=null;
    protected int $logLevel = 10;

    /**
     * @param array<string> $catalog
     */
    public function __construct(array $catalog = null)
    {
        $catalog = $catalog ?? [
            'Rindow\Math\Matrix\Drivers\MatlibFFI\MatlibFFI',
            'Rindow\Math\Matrix\Drivers\MatlibExt\MatlibExt',
        ];
        $this->catalog = $catalog;
    }

    protected function logging(int $level, string $message) : void
    {
        if($level > $this->logLevel) {
            return;
        }
        echo $message."\n";
    }

    public function select(int $verbose=null) : Service
    {
        $verbose ??= 0;
        $this->logLevel = 10 - $verbose;

        if($this->recommended) {
            return $this->recommended;
        }
        $recommended = null;
        $highestLevel = 0;
        foreach ($this->catalog as $name) {
            if(class_exists($name)) {
                $service = new $name();
                if(!($service instanceof Service)) {
                    throw new LogicException('Not service class: '.$name);
                }
                $level = $service->serviceLevel(verbose:$verbose);
                $this->logging(0, 'Service '.$name.' is level '.$level);
                if($level>$highestLevel) {
                    $highestLevel = $level;
                    $recommended = $service;
                    $this->logging(0, 'Update recommend to '.$name);
                }
            } else {
                $this->logging(1,'Service Not found: '.$name);
            }
        }
        if($highestLevel<=Service::LV_BASIC) {
            $recommended = new MatlibPhp();
        }
        if($recommended==null) {
            throw new RuntimeException('Service not found');
        }
        $this->recommended = $recommended;
        return $recommended;
    }
}