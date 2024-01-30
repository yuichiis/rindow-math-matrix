<?php
namespace RindowTest\Math\Matrix\NDArrayCLExtModeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibExt;
use Rindow\Math\Matrix\Drivers\Service;

if(!class_exists('RindowTest\Math\Matrix\NDArrayCLTest\NDArrayCLTest')) {
    require_once __DIR__.'/NDArrayCLTest.php';
}
use RindowTest\Math\Matrix\NDArrayCLTest\NDArrayCLTest as ORGTest;

/**
 * @requires extension rindow_opencl
 */
class NDArrayCLExtModeTest extends ORGTest
{
    public function setUp() : void
    {
        $this->service = new MatlibExt();
        if($this->service->serviceLevel()<Service::LV_ADVANCED) {
            throw new \Exception("the service is not Advanced.");
        }
    }
}
