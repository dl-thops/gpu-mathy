<!DOCTYPE html>
<html style="font-size: 16px;">
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta charset="utf-8">
    <meta name="keywords" content="">
    <meta name="description" content="">
    <meta name="page_type" content="np-template-header-footer-from-plugin">
    <title>Output</title>
    <link rel="stylesheet" href="nicepage.css" media="screen">
<link rel="stylesheet" href="Output.css" media="screen">
    <script class="u-script" type="text/javascript" src="jquery-1.9.1.min.js" defer=""></script>
    <script class="u-script" type="text/javascript" src="nicepage.js" defer=""></script>
    <meta name="generator" content="Nicepage 3.15.1, nicepage.com">
    <link id="u-theme-google-font" rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:100,100i,300,300i,400,400i,500,500i,700,700i,900,900i|Open+Sans:300,300i,400,400i,600,600i,700,700i,800,800i">
    
    
    <script type="application/ld+json">{
		"@context": "http://schema.org",
		"@type": "Organization",
		"name": "",
		"url": "index.php"
}</script>
    <meta property="og:title" content="Output">
    <meta property="og:type" content="website">
    <meta name="theme-color" content="#478ac9">
    <link rel="canonical" href="index.php">
    <meta property="og:url" content="index.php">
  </head>
  <body class="u-body"><header class="u-clearfix u-header u-palette-5-base u-sticky u-header" id="sec-cd89"><nav class="u-menu u-menu-dropdown u-offcanvas u-menu-1">
        <div class="menu-collapse" style="font-size: 1.3125rem; letter-spacing: 0px;">
          <a class="u-button-style u-custom-active-color u-custom-hover-color u-custom-left-right-menu-spacing u-custom-padding-bottom u-custom-text-active-color u-custom-text-color u-custom-text-hover-color u-custom-top-bottom-menu-spacing u-nav-link u-text-active-palette-1-base u-text-hover-palette-2-base" href="#">
            <svg><use xlink:href="#menu-hamburger"></use></svg>
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><symbol id="menu-hamburger" viewBox="0 0 16 16" style="width: 16px; height: 16px;"><rect y="1" width="16" height="2"></rect><rect y="7" width="16" height="2"></rect><rect y="13" width="16" height="2"></rect>
</symbol>
</defs></svg>
          </a>
        </div>
        <div class="u-custom-menu u-nav-container">
          <ul class="u-nav u-unstyled u-nav-1"><li class="u-nav-item"><a class="u-active-grey-60 u-button-style u-hover-grey-40 u-nav-link u-text-active-white u-text-hover-grey-15" href="Demo.html" style="padding: 36px 48px;">Demo</a>
</li><li class="u-nav-item"><a class="u-active-grey-60 u-button-style u-hover-grey-40 u-nav-link u-text-active-white u-text-hover-grey-15" href="About.html" style="padding: 36px 48px;">About</a>
</li><li class="u-nav-item"><a class="u-active-grey-60 u-button-style u-hover-grey-40 u-nav-link u-text-active-white u-text-hover-grey-15" href="Contact.html" style="padding: 36px 48px;">Contact</a>
</li></ul>
        </div>
        <div class="u-custom-menu u-nav-container-collapse">
          <div class="u-black u-container-style u-inner-container-layout u-opacity u-opacity-95 u-sidenav">
            <div class="u-sidenav-overflow">
              <div class="u-menu-close"></div>
              <ul class="u-align-center u-nav u-popupmenu-items u-unstyled u-nav-2"><li class="u-nav-item"><a class="u-button-style u-nav-link" href="Demo.html" style="padding: 36px 48px;">Demo</a>
</li><li class="u-nav-item"><a class="u-button-style u-nav-link" href="About.html" style="padding: 36px 48px;">About</a>
</li><li class="u-nav-item"><a class="u-button-style u-nav-link" href="Contact.html" style="padding: 36px 48px;">Contact</a>
</li></ul>
            </div>
          </div>
         <div class="u-black u-menu-overlay u-opacity u-opacity-70"></div>
        </div>
      </nav></header>


<?php
$infile = fopen("input.txt","w");
fwrite($infile,$_POST["input_code"]);
fclose($infile);
$ret = exec('./a.out input.txt output.txt');
?>

<script src="https://cdn.jsdelivr.net/gh/google/code-prettify@master/loader/run_prettify.js"></script>
<link href="prism.css" rel="stylesheet"></link>
<link href="tinku.css" rel="stylesheet"></link>

<style>
.wrapper{
  width: 100%;
  height: 100%;
  text-align: center;
  margin-top:10px;
}
.btn-copy{
  background-color: #38AFDD;
  border: transparent;
  border-bottom: 2px solid #0086B7;
  border-radius: 2px;
  padding: 10px;
  min-width: 100px;
  color: #fff;
}
.btn-copy:hover, .btn-copy:focus{
  background-color: #48A1C1;
  border-bottom: 2px solid #38AFDD;
  /*transition cross browser*/
  transition: all .3s ease-in;
  -webkit-transition: all .3s ease-in;
  -moz-transition:all .3s ease-in;
  -o-transition: all .3s ease-in;
}
</style>

<script>
function setClipboard() {
    fetch('output.txt')
    .then(response => response.text())
    .then(data => {
      var tempInput = document.createElement("input");
      tempInput.style = "position: absolute; left: -1000px; top: -1000px";
      tempInput.value = data;
      document.body.appendChild(tempInput);
      tempInput.select();
      document.execCommand("copy");
      document.body.removeChild(tempInput);
    }
)};
</script>

<div class="wrapper">
  <input type="hidden" id="input-url" value="Copied!">
  <button class="btn-copy" onclick="setClipboard()">Copy</button>
</div>

<div class="mcenter">
<div class="tcenter">
<pre class="prettyprint">
<?php
  $fn = fopen("output.txt","r");
  echo "<br><br>";
  while(! feof($fn))  {
        $result = fgets($fn);
        echo "&nbsp&nbsp&nbsp";
        echo htmlspecialchars($result);
  }

  fclose($fn);
?>
</pre>
</div>
</div>

    
    <footer class="u-align-center u-clearfix u-footer u-grey-80 u-footer" id="sec-e821"><div class="u-clearfix u-sheet u-sheet-1">
        <p class="u-small-text u-text u-text-grey-15 u-text-variant u-text-1" spellcheck="false">Site developed and maintained by<br>Praveen Gorre<br>Viswanath Tadi<br>
        </p>
      </div></footer>
    
  </body>
</html>
