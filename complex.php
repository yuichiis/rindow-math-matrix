<?php
include __DIR__.'/vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use function Rindow\Math\Matrix\C;
$phpmode = new Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp();
$mophp = new Rindow\Math\Matrix\MatrixOperator(service:$phpmode);
$mo = new Rindow\Math\Matrix\MatrixOperator();


$a = $mo->array([C(10,i:20),C(-1,i:-2)],dtype:NDArray::complex64);
//var_dump($a[0]);
//foreach($a as $v) {
//    var_dump($v);
//}
echo $mo->toString($a)."\n";

echo $mo->toString($a,format:'%f')."\n";

$a = $mo->complex(1,-2);
echo "$a\n";
$a = $mo->complex(imag:-2);
echo "$a\n";
