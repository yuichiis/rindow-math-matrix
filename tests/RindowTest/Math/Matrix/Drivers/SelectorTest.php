<?php
namespace RindowTest\Math\Matrix\Drivers\SelectorTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\MatlibExt;
use Rindow\Math\Matrix\Drivers\MatlibFFI;
use Rindow\Math\Matrix\Drivers\MatlibPhp;


use Rindow\Math\Matrix\Drivers\Selector;
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

class Test extends TestCase
{
    public function newSelector($catalog=null)
    {
        return new Selector($catalog);
    }

    public function testDefault()
    {
        $ext = new MatlibExt();
        $ffi = new MatlibFFI();
        $php = new MatlibPhp();
        $selector = $this->newSelector();
        $service = $selector->select();
        if($ffi->serviceLevel()<=Service::LV_BASIC && 
            $ext->serviceLevel()>Service::LV_BASIC ) {
            $this->assertInstanceOf(MatlibExt::class,$service);
        } elseif($service->serviceLevel()>=Service::LV_ADVANCED) {
            $this->assertInstanceOf(MatlibFFI::class,$service);
        } else {
            $this->assertInstanceOf(MatlibPhp::class,$service);
        }
    }

    public function testCatalog()
    {
        $ext = new MatlibExt();
        $ffi = new MatlibFFI();
        $php = new MatlibPhp();
        if($ext->serviceLevel()>$ffi->serviceLevel()) {
            $truesrv = $ext;
        } else {
            $truesrv = $ffi;
        }
        if($truesrv->serviceLevel()<=Service::LV_BASIC) {
            $truesrv = $php;
        }
        $catalog = [MatlibFFI::class,MatlibExt::class,MatlibPhp::class];
        $selector = $this->newSelector($catalog);
        $service = $selector->select();
        $this->assertInstanceOf(get_class($truesrv),$service);
    }
}
