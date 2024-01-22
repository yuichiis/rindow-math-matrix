<?php
namespace RindowTest\Math\Matrix\MatrixOperatorPhpModeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibPhp;
use Rindow\Math\Matrix\Drivers\Service;

if(!class_exists('RindowTest\Math\Matrix\MatrixOperatorTest\Test')) {
    require_once __DIR__.'/MatrixOperatorTest.php';
}
use RindowTest\Math\Matrix\MatrixOperatorTest\Test as ORGTest;

class Test extends ORGTest
{
    public function newMatrixOperator()
    {
        $service = new MatlibPhp();
        $mo = new MatrixOperator(service:$service);
        if($service->serviceLevel()!=Service::LV_BASIC) {
            throw new \Exception("the service is invalid.");
        }
        return $mo;
    }
}
