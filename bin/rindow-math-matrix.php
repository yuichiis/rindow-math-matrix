<?php
$paths = [
    __DIR__.'/../../../autoload.php',
    __DIR__.'/../vendor/autoload.php',
];
foreach($paths as $path) {
    if(file_exists($path)) {
        include_once $path;
        break;
    }
}
use Rindow\Math\Matrix\MatrixOperator;

$verbose = null;
if($argc>2) {
    if($argv[0]=='-v') {
        $verbose = 1;
    }
}
$mo = new MatrixOperator(verbose:$verbose);
echo $mo->service()->info();
