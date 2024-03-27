<?php
namespace RindowTest\Math\Matrix\DebugTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\Drivers\Selector;

class DebugTest extends TestCase
{
    public function newMatrixOperator()
    {
        $mo = new MatrixOperator();
        //if($service->serviceLevel()<Service::LV_ADVANCED) {
        //    throw new \Exception("the service is not Advanced.");
        //}
        return $mo;
    }

    public function testCreateFromArray()
    {
        $mo = $this->newMatrixOperator();
        echo $mo->service()->info()."\n";
        $this->assertTrue(true);
    }
}