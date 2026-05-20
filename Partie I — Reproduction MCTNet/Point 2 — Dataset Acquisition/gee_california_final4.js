// ============================================================
// CALIFORNIA FINAL v4 - 5 ZONES
// Même astuce : safetyImage masquée ajoutée à chaque collection
// ============================================================

var cal1 = ee.Geometry.Rectangle([-122.20, 39.00, -121.80, 39.40]);
var cal2 = ee.Geometry.Rectangle([-120.10, 36.80, -119.70, 37.20]);
var cal3 = ee.Geometry.Rectangle([-119.95, 36.55, -119.55, 36.95]);
var cal4 = ee.Geometry.Rectangle([-119.40, 35.20, -119.00, 35.60]);
var cal5 = ee.Geometry.Rectangle([-122.10, 39.10, -121.70, 39.50]);

var calAll = cal1.union(cal2).union(cal3).union(cal4).union(cal5);
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

function maskClouds(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0)
                .and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask);
}

var safetyImage = ee.Image.constant([0,0,0,0,0,0,0,0,0,0])
  .rename(bands).toFloat()
  .set('system:time_start', ee.Date('2021-01-01').millis())
  .set('CLOUDY_PIXEL_PERCENTAGE', 0)
  .selfMask();

function getComposite(startDay) {
  var start = ee.Date('2021-01-01').advance(startDay, 'day');
  var end = start.advance(10, 'day');
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(calAll)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskClouds)
    .select(bands)
    .map(function(img) { return img.toFloat(); });
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

var agMask = cropland.gte(1).and(cropland.lte(61))
  .or(cropland.eq(69))
  .or(cropland.eq(75))
  .or(cropland.eq(204));

var label = ee.Image(5).toFloat().rename('label');
label = label.where(cropland.eq(3),   0);  // Rice
label = label.where(cropland.eq(36),  1);  // Alfalfa
label = label.where(cropland.eq(69),  2);  // Grapes
label = label.where(cropland.eq(75),  3);  // Almonds
label = label.where(cropland.eq(204), 4);  // Pistachios

// ---- ESA WORLDCOVER v200 ----
var worldCover   = ee.Image('ESA/WorldCover/v200/2021').select('Map');
var croplandMask = worldCover.eq(40);

var combinedMask = agMask.and(confMask).and(croplandMask);

var dataset = timeSeries
  .addBands(label)
  .updateMask(combinedMask);

// ---- EXPORTS 5 ZONES ----
var zones = [cal1, cal2, cal3, cal4, cal5];
var names = ['California_Z1', 'California_Z2', 'California_Z3', 'California_Z4', 'California_Z5'];

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
