<?php
include __DIR__.'/vendor/autoload.php';

use Rindow\Math\Matrix\PhpMath;
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$math = new PhpMath();

$x = $mo->zeros([2,3,4]);
$x = $mo->arange($x->size(),dtype:NDArray::int32)->reshape($x->shape());
$sourceShape = $mo->array($x->shape(),NDArray::int32);
$perm = $mo->array(range(count($sourceShape)-1,0,-1));
$perm = $mo->array([2,0,1]);
$y = $mo->zeros(array_map(fn($dim)=>$x->shape()[$perm[$dim]],range(0,count($x->shape())-1)),$x->dtype());
echo "sourceShape=".$mo->toString($sourceShape)."\n";
echo "perm=".$mo->toString($perm)."\n";
$math->transpose(
    $x->buffer(),$x->offset(),
    $y->buffer(),$y->offset(),
    $sourceShape->buffer(),$perm->buffer(),
);
echo $mo->toString($x,null,true)."\n";
echo $mo->toString($y,null,true)."\n";
echo "x.shape=[".implode(',',$x->shape())."]\n";
echo "y.shape=[".implode(',',$y->shape())."]\n";
$a = $mo->array([
    [[1,2,3],[4,5,6]],
    [[7,8,9],[10,11,12]],
    [[13,14,15],[16,17,18]],
    [[19,20,21],[22,23,24]],
]);
echo $mo->toString($mo->transpose($a)); 
