import assert from 'assert'
import { readFileSync, writeFileSync } from 'fs'
import path from 'path'

import { AppData } from '../src/appdata'
import { testPattern } from './helpers'

describe("AppData", function () {
  const fileExt = '.mid'
  const dataPath = path.join(__dirname, 'fixtures')
  const appData = new AppData(dataPath, fileExt)
  it("checks constructor", function() {
    assert.ok(appData._rootPath === dataPath + '/.data')
    assert.ok(appData._fileExt === fileExt)
    assert.ok(appData._indexPath === dataPath + '/.data/.index.json')
  })
  it("check factory data", async function () {
    const expectedName = 'factory1'
    const expectedPath = process.cwd() + '/tests/fixtures/.data/factory/factory1.mid'
    assert.ok(appData._indexData[expectedName] = {name: expectedName, path: expectedPath, parent: 'factory'})
  })
  it("check user data", function () {
    const expectedName = 'user1'
    const expectedPath = process.cwd() + '/tests/fixtures/.data/factory/user1.mid'
    assert.ok(appData._indexData[expectedName] = {name: expectedName, path: expectedPath, parent: 'user'})
  })
  it("saves index", function() {
    writeFileSync(appData._indexPath, '{}', 'utf-8')
    const data = readFileSync(appData._indexPath, 'utf-8')
    const gotIndexdata = JSON.parse(data)
    assert.ok(JSON.stringify(gotIndexdata) === '{}')

    appData.saveIndex()
    const newData = readFileSync(appData._indexPath, 'utf-8')
    const gotNewIndexdata = JSON.parse(newData)
    const expectedIndexData = appData._indexData
    for (const [key, value] of Object.entries(expectedIndexData)) {
      assert.ok(value.name) == gotNewIndexdata[key].name
      assert.ok(value.path) == gotNewIndexdata[key].path
      assert.ok(value.parent) == gotNewIndexdata[key].parent
    }
  })
  it("saves patterns", async function() {
    const beforeData = readFileSync(appData._indexPath, 'utf-8')
    const beforeIndexData = JSON.parse(beforeData)

    const expectedName = 'test1'
    assert.ok(appData._indexData[expectedName] === undefined)

    const [onsets, velocities, offsets] = await testPattern()
    appData.savePattern(onsets, velocities, offsets, 'test1')
    
    const gotFileMeta = appData._indexData[expectedName]
    assert.ok(gotFileMeta.name === expectedName)
    assert.ok(gotFileMeta.parent === 'user')
    assert.ok(gotFileMeta.path === process.cwd() + '/tests/fixtures/.data/user/test1.mid')

    // reset in-memory index data
    appData._indexData = beforeIndexData
  })
})