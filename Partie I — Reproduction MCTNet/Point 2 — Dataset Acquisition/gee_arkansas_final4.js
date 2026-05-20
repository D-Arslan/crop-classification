// ============================================================
// ARKANSAS FINAL v4 - 5 ZONES
// Pas de ee.Algorithms.If
// Astuce : on ajoute une image de zéros masquée à chaque
// collection pour qu'elle ne soit jamais vide.
// median() ignorera les zéros masqués sauf si c'est la seule image.
// ============================================================

var ark1 = ee.Geometry.Rectangle([-91.75, 34.40, -91.25, 34.80]);
var ark2 = ee.Geometry.Rectangle([-92.10, 33.80, -91.60, 34.20]);
var ark3 = ee.Geometry.Rectangle([-91.65, 34.60, -91.15, 35.00]);
var ark4 = ee.Geometry.Rectangle([-90.90, 35.60, -90.40, 36.00]);
var ark5 = ee.Geometry.Rectangle([-91.80, 33.40, -91.30, 33.80]);

var arkAll = ark1.union(ark2).union(ark3).union(ark4).union(ark5);
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

function maskClouds(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0)
                .and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask);
}

// Image "filet de sécurité" : zéros avec les bonnes bandes et une date bidon
// Elle est MASQUÉE partout sauf si la collection est vide
// (quand c'est la seule image, unmask(0) la ramène à 0)
var safetyImage = ee.Image.constant([0,0,0,0,0,0,0,0,0,0])
  .rename(bands).toFloat()
  .set('system:time_start', ee.Date('2021-01-01').millis())
  .set('CLOUDY_PIXEL_PERCENTAGE', 0)
  .selfMask(); // masque tout (tous les pixels = 0 → masqués)

// ---- FONCTION COMPOSITE ----
function getComposite(startDay) {
  var start = ee.Date('2021-01-01').advance(startDay, 'day');
  var end = start.advance(10, 'day');
  
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(arkAll)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskClouds)
    .select(bands)
    .map(function(img) { return img.toFloat(); });
  
  // Ajouter safetyImage → collection jamais vide → median() a toujours 10 bandes
  // Si vraies images existent : median les utilise (safety est masquée, ignorée)
  // Si aucune vraie image : median = safety (masquée) → unmask(0) → zéros
  var safeColl = collection.merge(ee.ImageCollection([safetyImage]));
  
  return safeColl.median().unmask(0).toFloat();
}

// ---- 36 COMPOSITES EXPLICITES ----
var t0  = getComposite(0);   var t1  = getComposite(10);
var t2  = getComposite(20);  var t3  = getComposite(30);
var t4  = getComposite(40);  var t5  = getComposite(50);
var t6  = getComposite(60);  var t7  = getComposite(70);
var t8  = getComposite(80);  var t9  = getComposite(90);
var t10 = getComposite(100); var t11 = getComposite(110);
var t12 = getComposite(120); var t13 = getComposite(130);
var t14 = getComposite(140); var t15 = getComposite(150);
var t16 = getComposite(160); var t17 = getComposite(170);
var t18 = getComposite(180); var t19 = getComposite(190);
var t20 = getComposite(200); var t21 = getComposite(210);
var t22 = getComposite(220); var t23 = getComposite(230);
var t24 = getComposite(240); var t25 = getComposite(250);
var t26 = getComposite(260); var t27 = getComposite(270);
var t28 = getComposite(280); var t29 = getComposite(290);
var t30 = getComposite(300); var t31 = getComposite(310);
var t32 = getComposite(320); var t33 = getComposite(330);
var t34 = getComposite(340); var t35 = getComposite(350);

var timeSeries = t0.addBands(t1).addBands(t2).addBands(t3)
  .addBands(t4).addBands(t5).addBands(t6).addBands(t7)
  .addBands(t8).addBands(t9).addBands(t10).addBands(t11)
  .addBands(t12).addBands(t13).addBands(t14).addBands(t15)
  .addBands(t16).addBands(t17).addBands(t18).addBands(t19)
  .addBands(t20).addBands(t21).addBands(t22).addBands(t23)
  .addBands(t24).addBands(t25).addBands(t26).addBands(t27)
  .addBands(t28).addBands(t29).addBands(t30).addBands(t31)
  .addBands(t32).addBands(t33).addBands(t34).addBands(t35);

print('Nombre de bandes:', timeSeries.bandNames().size());

// ---- CDL 2021 ----
var cdl_full   = ee.Image('USDA/NASS/CDL/2021');
var cropland   = cdl_full.select('cropland');
var confidence = cdl_full.select('confidence');
var confMask   = confidence.gte(95);

var agMask = cropland.gte(1).and(cropland.lte(61));

var label = ee.Image(4).toFloat().rename('label');
label = label.where(cropland.eq(1), 0);   // Corn
label = label.where(cropland.eq(2), 1);   // Cotton
label = label.where(cropland.eq(3), 2);   // Rice
label = label.where(cropland.eq(5), 3);   // Soybeans

// ---- ESA WORLDCOVER v200 ----
var worldCover   = ee.Image('ESA/WorldCover/v200/2021').select('Map');
var croplandMask = worldCover.eq(40);

var combinedMask = agMask.and(confMask).and(croplandMask);

var dataset = timeSeries
  .addBands(label)
  .updateMask(combinedMask);

// ---- EXPORTS 5 ZONES ----
var zones = [ark1, ark2, ark3, ark4, ark5];
var names = ['Arkansas_Z1', 'Arkansas_Z2', 'Arkansas_Z3', 'Arkansas_Z4', 'Arkansas_Z5'];

for (var z = 0; z < zones.length; z++) {
  var samples = dataset.clip(zones[z]).sample({
    region    : zones[z],
    scale     : 30,
    numPixels : 10000,
    geometries: false,
    seed      : 42,
    tileScale : 16
  });
  print(names[z] + ' - Samples:', samples.size());
  Export.table.toDrive({
    collection     : samples,
    description    : names[z] + '_final',
    folder         : 'GEE_CropMapping',
    fileNamePrefix : names[z] + '_final',
    fileFormat     : 'CSV'
  });
}
