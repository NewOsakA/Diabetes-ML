# Maintaining the project wiki

The published guide is at [github.com/NewOsakA/diabetes-ml/wiki](https://github.com/NewOsakA/diabetes-ml/wiki). Its editable Markdown and image source is mirrored in `docs/wiki/` on the experiment branch.

GitHub stores a wiki in a separate Git repository. Changing these mirrored files does not automatically publish a wiki update. The two scientific experiments also remain on their existing branches; wiki publication does not merge them.

## Edit and preview

1. Edit the relevant page in `docs/wiki/`.
2. Keep result claims tied to a named experiment, evaluation method and committed evidence file.
3. For revised plots or diagrams, run `.venv/bin/python docs/build_wiki_visuals.py` from the project root. The script uses committed result tables and does not retrain models.
4. Check that all page links have matching Markdown files and every image URL names a file in `docs/wiki/assets/`.

The wiki uses static image diagrams so readers do not depend on an embedded diagram-rendering service. All charts have text alternatives and accompanying explanations.

## Publish an update

From the main project folder, clone the wiki once into the Git-ignored publishing folder:

```bash
git clone https://github.com/NewOsakA/diabetes-ml.wiki.git .wiki-publish
```

For subsequent updates, first run `git -C .wiki-publish pull --ff-only`. Copy the prepared content, inspect the change, then publish it:

```bash
.venv/bin/python -c "import shutil; shutil.copytree('docs/wiki', '.wiki-publish', dirs_exist_ok=True)"
git -C .wiki-publish add .
git -C .wiki-publish diff --cached --check
git -C .wiki-publish diff --cached --stat
git -C .wiki-publish commit -m 'Update beginner project guide'
git -C .wiki-publish push
```

This copy preserves the publishing folder's `.git` directory. It does not remove obsolete remote pages; review any intentional deletion separately. Commit and push the mirrored source changes to the project repository too, so maintainers can trace them.

Finally, open the published pages and check navigation, tables, diagram images and any changed numerical claims. Saving a browser preview records what readers actually see.
