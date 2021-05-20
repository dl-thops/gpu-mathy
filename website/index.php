<!DOCTYPE html>
<html style="font-size: 16px;">
	<?php
	exec('rm output.txt');
	exec('rm input.txt');
	?>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta charset="utf-8">
    <meta name="keywords" content="Summary, Exam​​ples">
    <meta name="description" content="">
    <meta name="page_type" content="np-template-header-footer-from-plugin">
    <title>Demo</title>
    <link rel="stylesheet" href="nicepage.css" media="screen">
<link rel="stylesheet" href="Demo.css" media="screen">
    <script class="u-script" type="text/javascript" src="jquery-1.9.1.min.js" defer=""></script>
    <script class="u-script" type="text/javascript" src="nicepage.js" defer=""></script>
    <meta name="generator" content="Nicepage 3.15.1, nicepage.com">
    <link id="u-theme-google-font" rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:100,100i,300,300i,400,400i,500,500i,700,700i,900,900i|Open+Sans:300,300i,400,400i,600,600i,700,700i,800,800i">
    
    
    
    <script type="application/ld+json">{
		"@context": "http://schema.org",
		"@type": "Organization",
		"name": "",
		"url": "index.html"
}</script>
    <meta property="og:title" content="Demo">
    <meta property="og:type" content="website">
    <meta name="theme-color" content="#478ac9">
    <link rel="canonical" href="index.html">
    <meta property="og:url" content="index.html">
  </head>
  <body data-home-page="https://website400824.nicepage.io/Demo.html?version=0c6968d3-8c01-41e2-b775-a208c74658a4" data-home-page-title="Demo" class="u-body"><header class="u-clearfix u-header u-palette-5-base u-sticky u-header" id="sec-cd89"><nav class="u-menu u-menu-dropdown u-offcanvas u-menu-1">
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
    <section class="u-align-left u-clearfix u-section-1" id="sec-165f">
      <div class="u-clearfix u-sheet u-sheet-1">
        <div class="u-clearfix u-custom-html u-custom-html-1">
          <style> /* Style inputs, select elements and textareas */
input[type=text], select, textarea{
  width: 100%;
  padding: 12px;
  border: 5px solid #888;
  border-radius: 4px;
  box-sizing: border-box;
  resize: vertical;
}
/* Style the label to display next to the inputs */
label {
  padding: 12px 12px 12px 0;
  display: inline-block;
}
/* Style the submit button */
input[type=submit] {
  background-color: #04AA6D;
  color: white;
  padding: 12px 20px 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  float: right;
}
/* Style the container */
.container {
  border-radius: 5px;
  background-color: #f2f2f2;
  padding: 20px;
}
/* Floating column for labels: 25% width */
.col-25 {
  float: left;
  width: 25%;
  margin-top: 6px;
}
/* Floating column for inputs: 75% width */
.col-75 {
  float: left;
  width: 75%;
  margin-top: 6px;
}
/* Clear floats after the columns */
.row:after {
  content: "";
  display: table;
  clear: both;
}
/* Responsive layout - when the screen is less than 600px wide, make the two columns stack on top of each other instead of next to each other */
@media screen and (max-width: 600px) {
  .col-25, .col-75, input[type=submit] {
    width: 100%;
    margin-top: 0;
  }
} </style>
          <div class="container">
            <form action="function.php" method="POST">
              <div class="row">
                <textarea id="subject" name="input_code" placeholder="Enter Mathy code here.." style="resize: none; height:400px"></textarea>
              </div>
              <div class="row" style="padding: 20px 0px 0px 0px;">
                <input type="submit" value="Compile">
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
    <section class="u-clearfix u-gradient u-section-2" id="sec-0d68">
      <div class="u-clearfix u-sheet u-sheet-1">
        <h1 class="u-text u-text-1" spellcheck="false">Examp<span style="font-size: 3.5rem;"></span>les
        </h1>
        <ul class="u-text u-text-2" spellcheck="false">
          <li>Calculating Mean of Array </li>
        </ul>
        <div class="u-align-left u-border-5 u-border-grey-50 u-container-style u-group u-radius-7 u-shape-round u-group-1" data-animation-name="zoomIn" data-animation-duration="1000" data-animation-delay="0" data-animation-direction="">
          <div class="u-container-layout u-container-layout-1">
            <p class="u-text u-text-3" data-animation-name="zoomIn" data-animation-duration="1000" data-animation-delay="0" data-animation-direction="">
              <span style="font-size: 1.625rem;">
                <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mean = sigma (a[i]/100) | 0 &lt;= i &lt; 100
              </span>
              <br>
              <br>
            </p>
          </div>
        </div>
        <ul class="u-text u-text-4" spellcheck="false">
          <li>Matrix Multiplication </li>
        </ul>
        <div class="u-align-left u-border-5 u-border-grey-50 u-container-style u-group u-radius-7 u-shape-round u-group-2" data-animation-name="zoomIn" data-animation-duration="1000" data-animation-delay="0" data-animation-direction="">
          <div class="u-container-layout u-valign-top u-container-layout-2">
            <p class="u-text u-text-5" data-animation-name="zoomIn" data-animation-duration="1000" data-animation-delay="0" data-animation-direction="">
              <span style="font-size: 1.625rem;">
                <br>forall (i) | 0 &lt;= i &lt;= 600 {<br>&nbsp;&nbsp;&nbsp; forall (j) | 0 &lt;= j &lt;= 800  { <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c[i][j] = sigma (a[i][k] * b[k][j]) | 0&lt;= k &lt;=200<br>&nbsp;&nbsp;&nbsp; }<br>}<br>
                <br>
              </span>
            </p>
          </div>
        </div>
        <h2 class="u-text u-text-6" spellcheck="false">Note<br>
        </h2>
        <p class="u-custom-font u-heading-font u-text u-text-7" spellcheck="false">1. ∀ <span class="u-text-grey-70">(forall)</span> , Σ <span class="u-text-grey-70">(sigma)</span> , Π <span class="u-text-grey-70">(product)</span> , √ <span class="u-text-grey-70">(sqrt)</span> , | <span class="u-text-grey-70">(where)</span> symbols are supported by the grammar.<br>
          <br>2. If the output page does not load <span class="u-text-grey-70">(probably due to syntax mistakes)</span> or you see the wrong output, refresh and try again. 
        </p>
      </div>
    </section>
    
    
    <footer class="u-align-center u-clearfix u-footer u-grey-80 u-footer" id="sec-e821"><div class="u-clearfix u-sheet u-sheet-1">
        <p class="u-small-text u-text u-text-grey-15 u-text-variant u-text-1" spellcheck="false">Site developed and maintained by<br>Praveen Gorre<br>Viswanath Tadi<br>
        </p>
      </div></footer>
    </span>
  </body>
</html>
