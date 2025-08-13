// market_fetch.js
const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");

(async () => {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error("Usage: node market_fetch.js <month> <year>");
        process.exit(1);
    }
    const [month, year] = args;

    const downloadPath = path.resolve(__dirname, "downloads");
    if (!fs.existsSync(downloadPath)) {
        fs.mkdirSync(downloadPath);
    }

    const browser = await puppeteer.launch({
        headless: true,
        args: [`--window-size=1920,1080`, `--disable-notifications`],
    });

    const page = await browser.newPage();
    await page._client().send("Page.setDownloadBehavior", {
        behavior: "allow",
        downloadPath: downloadPath,
    });

    console.log(`Opening Agmarknet for ${month}/${year}...`);
    await page.goto("https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReportpart2.aspx", { waitUntil: "networkidle2" });

    // Select Commodity
    await page.select("#ddlCommodity", "Cotton");

    // Select State
    await page.select("#ddlState", "Gujarat");

    // Select Year
    await page.select("#ddlYear", year);

    // Select Month
    await page.select("#ddlMonth", month);

    // Click "Go" button
    await page.click("#btnGo");
    await page.waitForTimeout(2000);

    // Click "Export to Excel"
    console.log("Downloading file...");
    await page.click("#btnExportToExcel");
    await page.waitForTimeout(5000);

    await browser.close();

    // Rename to .html
    const files = fs.readdirSync(downloadPath);
    const latestFile = files.sort((a, b) => {
        return fs.statSync(path.join(downloadPath, b)).mtime - fs.statSync(path.join(downloadPath, a)).mtime;
    })[0];

    const oldPath = path.join(downloadPath, latestFile);
    const newPath = path.join(downloadPath, "market_data.html");
    fs.renameSync(oldPath, newPath);
    console.log(`File saved as ${newPath}`);
})();
