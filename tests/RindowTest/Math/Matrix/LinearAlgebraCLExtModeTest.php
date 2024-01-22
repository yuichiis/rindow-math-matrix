<?php
namespace RindowTest\Math\Matrix\LinearAlgebraCLExtModeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibExt;
use Rindow\Math\Matrix\Drivers\Service;

if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraCLTest\Test')) {
    require_once __DIR__.'/LinearAlgebraCLTest.php';
}
use RindowTest\Math\Matrix\LinearAlgebraCLTest\Test as ORGTest;

/**
 * @requires extension rindow_openblas
 */
class Test extends ORGTest
{
    public function setUp() : void
    {
        $this->service = new MatlibExt();
        if($this->service->serviceLevel()<Service::LV_ACCELERATED) {
            $this->markTestSkipped("The service is not Accelerated.");
            throw new \Exception("The service is not Accelerated.");
        }
    }
}
