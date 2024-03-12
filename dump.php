<?php
include __DIR__.'/vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
$phpmode = new Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp();
$mophp = new Rindow\Math\Matrix\MatrixOperator(service:$phpmode);
$mo = new Rindow\Math\Matrix\MatrixOperator();

$a = $mo->array([0x12345678,0x55555555],dtype:NDArray::int32);
$data = $a->buffer()->dump();

echo bin2hex($data)."\n";
$u = unpack('L*',$data);
foreach($u as $value) {
    echo dechex($value)."\n";
}

$a = $mo->array([123.25, 345.125],dtype:NDArray::float32);
$data = $a->buffer()->dump();

echo bin2hex($data)."\n";
$u = unpack('g*',$data);
foreach($u as $value) {
    echo $value."\n";
}
echo $a->service()->info();
echo "\n";
$p = $mophp->zeros([2],dtype:NDArray::float32);
$p->buffer()->load($data);
echo $mophp->toString($p)."\n";
echo $p->service()->info();
echo "\n";

$data = $p->buffer()->dump();
$b = $mo->zeros([2],dtype:NDArray::float32);
$b->buffer()->load($data);
echo $mo->toString($b)."\n";
echo $b->service()->info();
echo "\n";

