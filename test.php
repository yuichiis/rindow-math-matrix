<?php
include __DIR__.'/vendor/autoload.php';
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();

$perm=null;
$shape=[4];
$shape=[2,3];
//$shape=[2,4,3,2];
//$perm=[0,2,1];
//$shape=[2,3,4];
$a = $mo->arange((int)array_product($shape),dtype:NDArray::int32)->reshape($shape);
//$a = $mo->arange((int)array_product($shape));

$b = $mo->transpose($a,perm:$perm);
//$b = $mo->transpose($a);

echo $mo->toString($b,'%3.0f',true)."\n";
echo '('.implode(',',$b->shape()).")\n";
