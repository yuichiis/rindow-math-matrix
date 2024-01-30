<?php
namespace RindowTest\Math\Matrix\MatrixOperatorExtModeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibExt;
use Rindow\Math\Matrix\Drivers\Service;

if(!class_exists('RindowTest\Math\Matrix\MatrixOperatorTest\MatrixOperatorTest')) {
    require_once __DIR__.'/MatrixOperatorTest.php';
}
use RindowTest\Math\Matrix\MatrixOperatorTest\MatrixOperatorTest as ORGTest;

/**
 * @requires extension rindow_openblas
 */
class MatrixOperatorExtModeTest extends ORGTest
{
    public function newMatrixOperator()
    {
        if(!extension_loaded('rindow_openblas')) {
            throw new \Exception("rindow_openblas is not loaded.");
        }
        $service = new MatlibExt();
        $mo = new MatrixOperator(service:$service);
        if($service->serviceLevel()<Service::LV_ADVANCED) {
            throw new \Exception("the service is not Advanced.");
        }
        return $mo;
    }
}
