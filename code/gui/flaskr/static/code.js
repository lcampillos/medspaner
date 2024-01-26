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
function mostrarTexto() {
  var select = document.getElementById("seleccionarTexto");
  var areaTexto = document.getElementById("text_input");
  var selectedValue = select.value;

  // Define los textos correspondientes a cada opción
  var textos = {
    empty: "",
    texto1: "MOTIVO DE CONSULTA:\nCansancio.\n\nHISTORIA CLÍNICA:\nMujer de 54 años.\n\n• Antecedentes personales:\n- Postmenopáusica.\n- Sin factores de riesgo cardiovascular.\n- Osteoporosis en tratamiento con denosumab.\n\n• Enfermedad actual:\n- Consulta por astenia intensa, poliuria y polidipsia.\n\nEXAMEN:\n\n• Pruebas complementarias:\n- Sedimento de orina con glucosuria 500mg/dl\n- Glucemia basal de 96mg/dl\n- Control analítico dentro de la normalidad.\n- Buenos niveles de calcio y fosfato en sangre.\n- Hemoglobina glicosilada A1c 5.0%\n- Test de sobrecarga oral de glucosa normal.\n- Muestra de orina de 24h: el resultado confirmó la glucosuria real.\n- Ecografía renal y de las vías urinarias: sin hallazgos de interés.\n\nDIAGNÓSTICO PRINCIPAL:\nGlucosuria Renal",
    texto2: "EudraCT Nº:  2020-001154-22\n\nTítulo público: Tocilizumab más terapia de base versus placebo más terapia de base en pacientes con neumonía grave por COVID-19\n\nTítulo científico: ESTUDIO ALEATORIZADO, DOBLE CIEGO, CONTROLADO CON PLACEBO Y MULTICÉNTRICO PARA EVALUAR LA SEGURIDAD Y LA EFICACIA DE TOCILIZUMAB EN PACIENTES CON NEUMONÍA GRAVE POR EL COVID-19\n\nIndicación pública: NEUMONÍA GRAVE POR EL COVID-19\n\nCriterios de inclusión:\n- Edad >=18 años\n- Hospitalizado con neumonía de la COVID-19 confirmada según los criterios de la OMS (incluida una RCP positiva de cualquier muestra; p. ej., respiratoria, sangre, orina, heces, otro líquido corporal) y mediante radiografía de tórax o TAC\n- SpO2 <=93 % o PaO2/FiO2 <300 mmhg\n\nCriterios de exclusión:\n- Reacciones alérgicas graves conocidas a TCZ u otros anticuerpos monoclonales\n- Infección por tuberculosis activa\n- Sospecha de infección bacteriana, fúngica, vírica u otra infección (además de COVID-19)\n- En opinión del investigador, la progresión a la muerte es inminente e inevitable en las próximas 24 horas, independientemente de la administración de tratamientos\n- Haber recibido fármacos antirrechazo orales o inmunomoduladores (incluido TCZ) en los últimos 6 meses\n- Embarazo o lactancia, o prueba de embarazo positiva en un examen previo a la dosis\n- Participación en otros ensayos clínicos con fármacos (con posible excepción de ensayos antivirales)\n- ALT o AST >10 x LSN\n- Tratamiento con un fármaco en investigación en las 5 semividas o 30 días\n- Cualquier afección médica grave o anomalía en pruebas analíticas clínicas.",
    texto3: "Cefazolina vs. Ciprofloxacino en la profilaxis de infecciones en pacientes cirróticos con sangrado digestivo.\nOBJETIVO: Determinar si el uso de Cefazolina como profilaxis antibiótica produce una disminución significativa de las infecciones en pacientes cirróticos con sangrado digestivo, comparado con Ciprofloxacino.\nMATERIAL Y MÉTODOS: Ensayo Clínico aleatorizado. Se incluyeron a pacientes cirróticos mayores de 18 años, con sangrado digestivo, que ingresaron entre Julio del 2008 a julio del 2010 a la Unidad de hemorragia Digestiva del HNERM, sin evidencia clínica ni de laboratorio de infección al momento del ingreso y que no hubieran recibido tratamiento antibiótico las últimas 2 semanas. A un grupo se le administró Ciprofloxacino 200 mg bid EV y al otro Cefazolina 1 g tid EV. x 7 días.\nRESULTADOS: Fueron incluidos 98 pacientes, 53 pacientes en el grupo de Cefazolina y 45 en el grupo de Ciprofloxacino. El promedio de edad fue 66 años, 61 % varones; 59,2 % tuvieron ascitis, la frecuencia de infecciones en la población total fue de 14,3% (14/98). El resangrado fue 8,1% y la mortalidad general 4,1%. No hubo diferencias significativas entre los grupos en relación a edad, sexo, estadio Child, ascitis, encefalopatía, ni promedio de Bilirrubina, TP, Creatinina y niveles de Albúmina. El grupo que usó Cefazolina tuvo 11,3 % de infecciones, comparado con el 17,8% de infecciones en el grupo que recibió Ciprofloxacino (p= 0,398) IC 95%. Cuando se excluyó del análisis los pacientes cirróticos Child A y aquellos sin ascitis, se encontró: 22,2 % de infecciones en el grupo de cefazolina y 26,9 % en el grupo de Ciprofloxacino (p=0,757 IC 95%). \nCONCLUSIÓN: no hubo diferencias significativas entre cefazolina y ciprofloxacino como profilaxis antibiótica en pacientes cirróticos sangrado digestivo.\n(Díaz Ferrer et al. 2011, PMID: 22476119)"
  };

  // Muestra el texto seleccionado en el área designada
  areaTexto.innerHTML = textos[selectedValue];
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
