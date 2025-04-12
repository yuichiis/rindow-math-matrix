<?php
ini_set('short_open_tag', '1');

date_default_timezone_set('UTC');
#ini_set('short_open_tag',true);
if(file_exists(__DIR__.'/../vendor/autoload.php')) {
    $loader = require_once __DIR__.'/../vendor/autoload.php';
    echo "autoload!!\n";
} else {
    $loader = require_once __DIR__.'/init_autoloader.php';
    echo "init_autoload!!\n";
}
$addpack = getenv('ADD_PACK');
$workingbranch = getenv('WORKING_BRANCH');
echo "addpack:$addpack\n";
echo "workingbranch:$workingbranch\n";
if(file_exists("$addpack/rindow-math-matrix-matlibffi-$workingbranch/composer.json")) {
    echo "addpack!!\n";
    $loader->addPsr4('Rindow\\Math\\Matrix\\Drivers\\MatlibFFI\\', "$addpack/rindow-math-matrix-matlibffi-$workingbranch/src");
    $loader->addPsr4('Rindow\\Math\\Buffer\\FFI\\', "$addpack/rindow-math-buffer-ffi-$workingbranch/src");
    $loader->addPsr4('Rindow\\Matlib\\FFI\\',   "$addpack/rindow-matlib-ffi-$workingbranch/src");
    $loader->addPsr4('Rindow\\OpenBLAS\\FFI\\', "$addpack/rindow-openblas-ffi-$workingbranch/src");
    $loader->addPsr4('Rindow\\OpenCL\\FFI\\',   "$addpack/rindow-opencl-ffi-$workingbranch/src");
    $loader->addPsr4('Rindow\\CLBlast\\FFI\\',  "$addpack/rindow-clblast-ffi/src");
}
#if(!class_exists('PHPUnit\Framework\TestCase')) {
#    include __DIR__.'/travis/patch55.php';
#}
