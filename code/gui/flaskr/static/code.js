/* Show loader during annotation process */
const nav = () => {
  var loader = document.getElementById('loader');
  loader.style.display='block';
  setTimeout(()=>loader.style.display = 'none', 100000);
}
/* This prevents loading the previous page when clicking on Back button  */
window.onpageshow = function(event) {
  if (event.persisted) {
  window.location.reload() 
  }
};
/* Get input text to annotate */
function getContent(){
  document.getElementById("my-textarea").value = document.getElementById("text_input").innerText;
}
/* Show download button */
var ann_text = document.getElementById("annotations").innerText;
var showDownloadButton = document.getElementById('show-after-results');
if (ann_text != '') {
  showDownloadButton.className = "show";
}
/* Load sample text */
function load_sample(){
  var sample_text = "La COVID-19 afecta de distintas maneras en función de cada persona. La mayoría de las personas que se contagian presentan síntomas de intensidad leve o moderada, y se recuperan sin necesidad de hospitalización.\nLos síntomas más habituales son los siguientes: fiebre, tos seca y cansancio.\nOtros síntomas menos comunes son los siguientes: Molestias y dolores, dolor de garganta, diarrea, conjuntivitis, dolor de cabeza, pérdida del sentido del olfato, erupciones cutáneas en los dedos de las manos o de los pies.\n(www.who.int)\n_____________________________________________________________________________________\n\nLa hipertensión y la enfermedad cardiovascular son más frecuentes en quienes evolucionan peor por COVID-19. Los pacientes mayores de 60 años, así como aquellos con enfermedad cardiovascular, deberían evitar especialmente la exposición al coronavirus, no automedicarse y consultar rápidamente ante la aparición de síntomas. El mecanismo de entrada del coronavirus a las células ha puesto en tela de juicio el uso de IECA.\n(Salazar et al. 2020, PMID: 32591283)\n_____________________________________________________________________________________\n\nTítulo público: Tocilizumab más terapia de base versus placebo más terapia de base en pacientes con neumonía grave por COVID-19\nIndicación pública: NEUMONÍA GRAVE POR EL COVID-19\nCriterios de inclusión:\n- Edad >=18 años\n- Hospitalizado con neumonía de la COVID-19 confirmada (incluida una RCP positiva de sangre, orina, heces, otro líquido corporal) y mediante radiografía de tórax o TAC\n(EudraCT nº 2020-001154-22)\n_____________________________________________________________________________________\n\nCefazolina vs. Ciprofloxacino en la profilaxis de infecciones en pacientes cirróticos con sangrado digestivo\nOBJETIVO: Determinar si el uso de Cefazolina como profilaxis antibiótica produce una disminución significativa de las infecciones en pacientes cirróticos con sangrado digestivo, comparado con Ciprofloxacino.\nMATERIAL Y MÉTODOS: Ensayo Clínico aleatorizado. Se incluyeron a pacientes cirróticos mayores de 18 años, con sangrado digestivo, que ingresaron entre Julio del 2008 a julio del 2010 a la Unidad de hemorragia Digestiva del HNERM, sin evidencia clínica ni de laboratorio de infección al momento del ingreso y que no hubieran recibido tratamiento antibiótico las últimas 2 semanas. A un grupo se le administró Ciprofloxacino 200 mg bid EV y al otro Cefazolina 1 g tid EV. x 7 días. \nRESULTADOS: Fueron incluidos 98 pacientes, 53 pacientes en el grupo de Cefazolina y 45 en el grupo de Ciprofloxacino. El promedio de edad fue 66 años, 61 % varones; 59,2 % tuvieron ascitis, la frecuencia de infecciones en la población total fue de 14,3% (14/98). El resangrado fue 8,1% y la mortalidad general 4,1%. No hubo diferencias significativas entre los grupos en relación a edad, sexo, estadio Child, ascitis, encefalopatía, ni promedio de Bilirrubina, TP, Creatinina y niveles de Albúmina. El grupo que usó Cefazolina tuvo 11,3 % de infecciones, comparado con el 17,8% de infecciones en el grupo que recibió Ciprofloxacino (p= 0,398) IC 95%. Cuando se excluyó del análisis los pacientes cirróticos Child A y aquellos sin ascitis, se encontró: 22,2 % de infecciones en el grupo de cefazolina y 26,9 % en el grupo de Ciprofloxacino (p=0,757 IC 95%).\nCONCLUSIÓN: no hubo diferencias significativas entre cefazolina y ciprofloxacino como profilaxis antibiótica en pacientes cirróticos sangrado digestivo.\n(Díaz Ferrer et al. 2011, PMID: 22476119)";
  document.getElementById("text_input").innerText = sample_text;
}
/* Download text to file */
function download(ann_text) {
  var filename = "anotado.ann";
  var ann_text = document.getElementById("annotations").innerText;
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(ann_text));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}


/* Function to hide title of broader entity when hovering on the inner nested entity */

var i;

for (i = 1; i < 1000; i++) {
  
  /* dynamic variable and attribute names */
  /* inner or nested entities */
  var inner_var = "inner"+i;
  var inner_att_var = "inner"+i+"att";

  /* flat or outer entities */
  var flat_var = "flat"+i;
  var flat_att_var = "flat"+i+"att";
  
  /* Create a function to extract data from each each pair of inner and nested entities */
  /* Use try ... catch ... because flat entities without inner entities stops running the code */
  s = "try { var " + inner_var + " = document.getElementById('" + inner_var + "'); var " + inner_att_var + " = " + inner_var + ".getAttribute('data-title'); var " + flat_var + " = document.getElementById('" + flat_var + "'); var " + flat_att_var + " = " + flat_var + ".getAttribute('data-title'); " + inner_var + ".addEventListener('mouseover', (event) => { " + flat_var + ".setAttribute('data-title', " + inner_att_var + "); " + inner_var + ".removeAttribute('data-title'); }); " + inner_var + ".addEventListener('mouseout', (event) => { " + flat_var + ".setAttribute('data-title'," + flat_att_var + "); " + inner_var + ".setAttribute('data-title', " + inner_att_var + "); }) } catch (e) { }";
  eval(s);
}
