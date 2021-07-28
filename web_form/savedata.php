<?php
	$file = fopen("user_data.txt","a");
	fwrite($file, $_POST["data"]."\n");
	fclose($file);
?>
