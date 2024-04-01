<?php
include __DIR__.'/vendor/autoload.php';

$mo = new Rindow\Math\Matrix\MatrixOperator();
$config = $mo->la()->getBlas()->getConfig();
var_dump($config);
$mode = $mo->la()->getBlas()->getParallel();
$modenames = [
    'SEQUENTIAL',
    'THREAD',
    'OPENMP',
];
var_dump('blas: '.$modenames[$mode]);
$mode = $mo->la()->getMath()->getParallel();
$modenames = [
    'SEQUENTIAL',
    'THREAD',
    'OPENMP',
];
var_dump('matlib: '.$modenames[$mode]);
