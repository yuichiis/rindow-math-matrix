<?php
if(version_compare(phpversion(),"8.0.0")>=0) {
    require_once __DIR__.'/NDArrayCLV8.php';
} else {
    require_once __DIR__.'/NDArrayCLV7.php';
}
