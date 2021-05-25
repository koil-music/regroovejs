import fs from "fs";
import glob from "glob";
import path from "path";

import { Pattern } from "./pattern";
import { readMidiFile, writeMidiFile } from "./midi";
import { pitchToIndexMap } from "./util";

interface FileMeta {
  name: string;
  parent: string;
  path: string;
}

interface IAppData {
  _rootPath: string;
  _indexData: Record<string, FileMeta>;
  factoryDir: string;
  userDir: string;
  _fileExt: string;
  _loadIndexFromJSON: (
    indexData: Record<string, Record<string, string>>
  ) => Record<string, FileMeta>;
  loadPattern: (name: string) => Promise<[Pattern, Pattern, Pattern]>;
  savePattern: (
    onsets: Pattern,
    velocities: Pattern,
    offsets: Pattern,
    name: string
  ) => void;
}

class AppData implements IAppData {
  _rootPath: string;
  _indexData: Record<string, FileMeta>;
  _fileExt: string;
  _indexPath: string;

  constructor(root: string, fileExt: string) {
    this._rootPath = root + "/.data";
    this._fileExt = fileExt;
    this._indexPath = this._rootPath + "/.index.json";
    let index = {};

    const data = fs.readFileSync(this._indexPath, "utf-8");
    const indexData = JSON.parse(data);
    index = this._loadIndexFromJSON(indexData);
    this._indexData = this._loadFactoryIndexData(index);
  }

  saveIndex(): void {
    fs.writeFileSync(this._indexPath, JSON.stringify(this._indexData));
  }

  _loadFactoryIndexData(
    index: Record<string, FileMeta>
  ): Record<string, FileMeta> {
    glob(
      this.factoryDir + this._fileExt,
      function (err, files: string[]) {
        if (err != null) throw err;
        for (const f of files) {
          const name = path.basename(f, this._fileExt);
          index[name] = { name: name, path: f, parent: "factory" };
        }
      }.bind(this)
    );
    return index;
  }

  _loadIndexFromJSON(
    indexData: Record<string, Record<string, string>>
  ): Record<string, FileMeta> {
    const index = {};
    for (const [key, value] of Object.entries(indexData)) {
      index[key] = { name: value.name, path: value.path };
    }
    return index;
  }

  _save(name: string, path: string): void {
    const data = { name: name, path: path, parent: "user" };
    this._indexData[name] = data;
  }

  get factoryDir(): string {
    return this._rootPath + "/factory/";
  }

  get userDir(): string {
    return this._rootPath + "/user/";
  }

  get data(): Record<string, FileMeta> {
    return this._indexData;
  }

  async savePattern(
    onsets: Pattern,
    velocities: Pattern,
    offsets: Pattern,
    name: string
  ): Promise<void> {
    const savePath = this.userDir + `${name}.mid`;
    this._save(name, savePath);
    await writeMidiFile(onsets, velocities, offsets, savePath);
  }

  async loadPattern(name: string): Promise<[Pattern, Pattern, Pattern]> {
    // TODO: Implement error handling if name doesn't exist
    const filePath = this._indexData[name].path;
    const pitchMapping = pitchToIndexMap();
    const [onsets, velocities, offsets] = await readMidiFile(
      filePath,
      pitchMapping
    );
    return [onsets, velocities, offsets];
  }
}

export { AppData, FileMeta };
