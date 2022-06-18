<?php
if(version_compare(phpversion(),"8.0.0")>=0) {
    require_once __DIR__.'/NDArrayPhpV8.php';
} else {
    require_once __DIR__.'/NDArrayPhpV7.php';
}
