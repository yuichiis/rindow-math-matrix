<?php
namespace RindowTest\Math\Matrix\MatrixOperatorTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\Drivers\Selector;

class MatrixOperatorTest extends TestCase
{
    public function newMatrixOperator()
    {
        $selector = new Selector();
        $service = $selector->select();
        $mo = new MatrixOperator(service:$service);
        if($service->serviceLevel()<Service::LV_ADVANCED) {
            throw new \Exception("the service is not Advanced.");
        }
        return $mo;
    }

    public function testCreateFromArray()
    {
        $mo = $this->newMatrixOperator();
        $this->assertTrue(true);
    }
}