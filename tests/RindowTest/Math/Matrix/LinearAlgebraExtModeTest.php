<?php
namespace RindowTest\Math\Matrix\LinearAlgebraExtModeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibExt;
use Rindow\Math\Matrix\Drivers\Service;

if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\LinearAlgebraTest')) {
    require_once __DIR__.'/LinearAlgebraTest.php';
}
use RindowTest\Math\Matrix\LinearAlgebraTest\LinearAlgebraTest as ORGTest;

/**
 * @requires extension rindow_openblas
 */
class LinearAlgebraExtModeTest extends ORGTest
{
    public function setUp() : void
    {
        $this->service = new MatlibExt();
        if($this->service->serviceLevel()<Service::LV_ADVANCED) {
            throw new \Exception("the service is not Advanced.");
        }
    }
}
