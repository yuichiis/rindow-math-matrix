<?php
namespace Rindow\Math\Matrix\Drivers;

class Selector 
{
    protected array $catalog;
    protected ?Service $recommended=null;

    public function __construct(array $catalog = null)
    {
        $catalog = $catalog ?? [
            MatlibFFI::class,
            MatlibExt::class,
        ];
        $this->catalog = $catalog;
    }

    public function select() : ?Service
    {
        if($this->recommended) {
            return $this->recommended;
        }
        $recommended = null;
        $highestLevel = 0;
        foreach ($this->catalog as $name) {
            if(class_exists($name)) {
                $service = new $name;
                $level = $service->serviceLevel();
                if($level>$highestLevel) {
                    $highestLevel = $level;
                    $recommended = $service;
                }
            }
        }
        if($highestLevel<=Service::LV_BASIC) {
            $recommended = new MatlibPhp();
        }
        $this->recommended = $recommended;
        return $recommended;
    }
}
