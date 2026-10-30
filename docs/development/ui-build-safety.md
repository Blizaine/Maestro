# UI build and deployment workflow

`npm run build` records the exact-byte hashes of the UI source and build configuration, runs TypeScript and Vite into a private staging directory, then validates the output before activation. The UI index is switched last. Existing hashed assets are retained; changed stable files and the previous index are backed up under an ignored `.maestro-build-backups-*` directory.

The input manifest includes files under `ui/src`, `ui/public`, and `ui/scripts`, plus top-level package, lock, TypeScript, Vite, lint, environment, and related build configuration files. Hashes use raw bytes, including line endings. The optional ignored `ui/.maestro-source-baseline.json` adds a reviewed-source check. Without a baseline, ordinary builds are allowed. With a baseline, source drift stops the build and lists changed paths.

## Cross-tree build

Build from the reviewed UI tree, then deploy that staged output only to a target whose complete inputs match the build receipt:

```powershell
cd <reviewed-tree>\ui
npm run build -- --stage-only --outDir <absolute-stage-directory>
node scripts\ui-build.mjs deploy --from <absolute-stage-directory> --ui <absolute-target-ui-directory>
node scripts\ui-build.mjs verify --ui <absolute-target-ui-directory>
```

The deploy command does not copy or reconcile source files. If target inputs differ, stop and synchronize them from the reviewed source first, then build again. The staged receipt and every output file are checked against the target before activation. Never use compiled output from a different UI source snapshot.

## Updating a reviewed baseline

When the baseline check reports drift, review every listed addition, removal, and change against the intended source update. Synchronize the reviewed UI tree first. Then explicitly replace the local ignored baseline only from that reviewed tree:

```powershell
node scripts\ui-build.mjs approve-baseline --from <absolute-reviewed-ui-directory>
```

This command succeeds only when the current UI input files exactly match the supplied reviewed tree. It does not approve or reset source differences. Do not delete the baseline to bypass a drift report.

## Checks

Run the fake-file safety regressions without compiling or publishing the application:

```powershell
node tests\ui\ui_build_safety.cjs
```

To inspect the currently served output against the local receipt and current source, run `npm run build:verify` from `ui/`.
