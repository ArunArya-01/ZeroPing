from __future__ import annotations

import logging
import os
import random
import sys
from functools import reduce
from pathlib import PurePosixPath
from typing import Any, Literal

import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.errors import LocalTokenNotFoundError


DATASET_REPO_ID = "aerotwin/aero-data"
HF_DATASET_PREFIX = "hf://datasets"
VALID_SPLITS: tuple[str, ...] = ("train", "rank", "final")

FLIGHTLIST_TEMPLATE = "flightlist_{split}.parquet"
FUEL_TEMPLATE = "fuel_{split}.parquet"
FLIGHTS_DIR_TEMPLATE = "flights_{split}"
FLIGHT_FILE_GLOB_TEMPLATE = "flights_{split}/*.parquet"

DEFAULT_SPLIT: Literal["train"] = "train"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_RANDOM_SEED = 42

FLIGHT_ID_COLUMNS: tuple[str, ...] = (
    "flight_id",
    "flightid",
    "id",
    "flight",
    "flight_number",
)
FLIGHT_PATH_COLUMNS: tuple[str, ...] = (
    "path",
    "file",
    "filepath",
    "file_path",
    "filename",
    "flight_file",
    "flight_path",
    "parquet_path",
)

LOGGER = logging.getLogger(__name__)
Split = Literal["train", "rank", "final"]


class AeroDataLoader:
    """Load AeroTwin dataset tables and flight parquet files from Hugging Face.

    Parameters
    ----------
    repo_id:
        Hugging Face dataset repository id. Defaults to
        ``"aerotwin/aero-data"``.
    token:
        Optional Hugging Face access token. When omitted, the loader relies on
        ``HF_TOKEN`` or an existing ``huggingface-cli login`` session.
    endpoint:
        Optional Hugging Face Hub endpoint, useful for private Hub mirrors.
    logger:
        Optional logger. When omitted, this module's logger is used.

    Notes
    -----
    All paths used for parquet reads are native ``hf://datasets/...`` paths.
    Returned data objects are Polars ``DataFrame`` instances.
    """

    def __init__(
        self,
        repo_id: str = DATASET_REPO_ID,
        token: str | None = None,
        endpoint: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.repo_id = repo_id.strip("/")
        self.token = token or os.getenv("HF_TOKEN")
        self.endpoint = endpoint
        self.logger = logger or LOGGER

        fs_options = self._hf_options()
        self.api = HfApi(endpoint=self.endpoint)
        self.fs = HfFileSystem(**fs_options)
        self._flight_files_cache: dict[Split, list[str]] = {}

    def get_flightlist(self, split: Split | str = DEFAULT_SPLIT) -> pl.DataFrame:
        """Return the flight list table for a split.

        Parameters
        ----------
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        polars.DataFrame
            The requested ``flightlist_<split>.parquet`` table.
        """

        split = self._validate_split(split)
        path = self._hf_path(FLIGHTLIST_TEMPLATE.format(split=split))
        self.logger.info("Loading flight list for split '%s' from %s", split, path)
        return self._load_parquet(path)

    def get_fuel_labels(self, split: Split | str = DEFAULT_SPLIT) -> pl.DataFrame:
        """Return the fuel labels table for a split.

        Parameters
        ----------
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        polars.DataFrame
            The requested ``fuel_<split>.parquet`` table.
        """

        split = self._validate_split(split)
        path = self._hf_path(FUEL_TEMPLATE.format(split=split))
        self.logger.info("Loading fuel labels for split '%s' from %s", split, path)
        return self._load_parquet(path)

    def list_flight_files(self, split: Split | str = DEFAULT_SPLIT) -> list[str]:
        """List parquet flight files for a split without reading their content.

        Parameters
        ----------
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        list[str]
            Sorted ``hf://`` paths for files under ``flights_<split>/``.
        """

        split = self._validate_split(split)
        if split in self._flight_files_cache:
            cached = self._flight_files_cache[split]
            self.logger.debug("Using cached list of %d flight files for split '%s'", len(cached), split)
            return cached[:]

        directory = FLIGHTS_DIR_TEMPLATE.format(split=split)
        self.logger.info("Listing flight files for split '%s' under %s", split, self._hf_path(directory))

        try:
            entries = self.api.list_repo_tree(
                repo_id=self.repo_id,
                path_in_repo=directory,
                recursive=True,
                repo_type="dataset",
                token=self.token,
            )
            files = [
                self._hf_path(entry.path)
                for entry in entries
                if getattr(entry, "path", "").endswith(".parquet")
            ]
        except Exception as exc:  # pragma: no cover - depends on remote Hub state
            self._raise_friendly_error(exc, self._hf_path(directory))

        normalized = sorted(self._normalize_hf_path(path) for path in files)
        self._flight_files_cache[split] = normalized
        self.logger.info("Found %d flight files for split '%s'", len(normalized), split)
        return normalized[:]

    def sample_flight_files(
        self,
        n: int = DEFAULT_SAMPLE_SIZE,
        split: Split | str = DEFAULT_SPLIT,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> list[str]:
        """Return a deterministic random sample of flight file paths.

        Parameters
        ----------
        n:
            Number of flight files to sample. If ``n`` exceeds the number of
            available files, all files are returned in deterministic shuffled
            order.
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.
        seed:
            Seed for reproducible sampling.

        Returns
        -------
        list[str]
            Sampled ``hf://`` flight parquet paths.
        """

        if n < 0:
            raise ValueError("n must be non-negative.")

        files = self.list_flight_files(split=split)
        if n == 0 or not files:
            return []

        rng = random.Random(seed)
        if n >= len(files):
            sampled = files[:]
            rng.shuffle(sampled)
            self.logger.warning("Requested %d files but only %d are available.", n, len(files))
            return sampled

        return rng.sample(files, n)

    def load_flight(self, path: str) -> pl.DataFrame:
        """Load a single flight parquet file.

        Parameters
        ----------
        path:
            Flight parquet path. Accepts a full ``hf://`` path, a dataset
            relative path such as ``flights_train/abc.parquet``, or a file name
            that already includes a split directory.

        Returns
        -------
        polars.DataFrame
            Flight time-series data.
        """

        hf_path = self._normalize_hf_path(path)
        self.logger.info("Loading flight from %s", hf_path)
        return self._load_parquet(hf_path)

    def load_flight_by_id(self, flight_id: str | int, split: Split | str = DEFAULT_SPLIT) -> pl.DataFrame:
        """Load a single flight by id.

        The loader first inspects ``flightlist_<split>.parquet`` for a matching
        id column. If the matching row contains a file path column, that path is
        used. Otherwise, it tries common ``flights_<split>/<id>.parquet`` style
        names and finally falls back to metadata-only file listing.

        Parameters
        ----------
        flight_id:
            Flight identifier to locate.
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        polars.DataFrame
            Flight time-series data for the requested id.
        """

        split = self._validate_split(split)
        flight_id_text = str(flight_id)
        self.logger.info("Loading flight id '%s' from split '%s'", flight_id_text, split)

        row = self._flightlist_row_for_id(flight_id_text, split)
        if row:
            path = self._path_from_flightlist_row(row, split)
            if path:
                return self.load_flight(path)

        for candidate in self._flight_id_path_candidates(flight_id_text, split):
            if self._path_exists(candidate):
                return self.load_flight(candidate)

        matches = self._matching_listed_flight_files(flight_id_text, split)
        if len(matches) == 1:
            return self.load_flight(matches[0])
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Flight id '{flight_id_text}' matched multiple files in split '{split}': {matches[:5]}"
            )

        raise FileNotFoundError(
            f"Could not locate flight id '{flight_id_text}' in split '{split}'. "
            "Check the id, split, and dataset file naming convention."
        )

    def load_sample_flights(self, n: int = DEFAULT_SAMPLE_SIZE, split: Split | str = DEFAULT_SPLIT) -> pl.DataFrame:
        """Load and concatenate a sample of flight files.

        Parameters
        ----------
        n:
            Number of flights to load.
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        polars.DataFrame
            Concatenated flight rows. A ``source_path`` column is added when it
            does not already exist so each row can be traced to its parquet file.
        """

        split = self._validate_split(split)
        paths = self.sample_flight_files(n=n, split=split)
        if not paths:
            self.logger.warning("No sampled flight files found for split '%s'.", split)
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        for path in paths:
            frame = self.load_flight(path)
            if "source_path" not in frame.columns:
                frame = frame.with_columns(pl.lit(path).alias("source_path"))
            frames.append(frame)

        self.logger.info("Concatenating %d sampled flight files for split '%s'", len(frames), split)
        return pl.concat(frames, how="diagonal_relaxed")

    def get_schema(self, split: Split | str = DEFAULT_SPLIT) -> pl.DataFrame:
        """Return schemas for the split's metadata tables and one flight file.

        Parameters
        ----------
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        polars.DataFrame
            Columns: ``split``, ``table``, ``path``, ``column``, and ``dtype``.
        """

        split = self._validate_split(split)
        schema_targets = {
            "flightlist": self._hf_path(FLIGHTLIST_TEMPLATE.format(split=split)),
            "fuel_labels": self._hf_path(FUEL_TEMPLATE.format(split=split)),
        }

        flight_files = self.list_flight_files(split=split)
        if flight_files:
            schema_targets["flight_example"] = flight_files[0]
        else:
            self.logger.warning("No flight files found for split '%s'; omitting flight schema.", split)

        rows: list[dict[str, Any]] = []
        for table, path in schema_targets.items():
            schema = self._read_schema(path)
            rows.extend(
                {
                    "split": split,
                    "table": table,
                    "path": path,
                    "column": column,
                    "dtype": str(dtype),
                }
                for column, dtype in schema.items()
            )

        return pl.DataFrame(rows)

    def dataset_summary(self, split: Split | str = DEFAULT_SPLIT) -> pl.DataFrame:
        """Return a compact summary for one dataset split.

        The summary avoids reading individual flight files. It counts metadata
        table rows with lazy parquet scans and counts flight files with Hugging
        Face filesystem metadata.

        Parameters
        ----------
        split:
            Dataset split. Must be one of ``"train"``, ``"rank"``, or
            ``"final"``.

        Returns
        -------
        polars.DataFrame
            One row per split resource with row, file, and column counts.
        """

        split = self._validate_split(split)
        flightlist_path = self._hf_path(FLIGHTLIST_TEMPLATE.format(split=split))
        fuel_path = self._hf_path(FUEL_TEMPLATE.format(split=split))
        flight_files = self.list_flight_files(split=split)

        rows = [
            {
                "split": split,
                "resource": "flightlist",
                "path": flightlist_path,
                "rows": self._row_count(flightlist_path),
                "files": None,
                "columns": len(self._read_schema(flightlist_path)),
            },
            {
                "split": split,
                "resource": "fuel_labels",
                "path": fuel_path,
                "rows": self._row_count(fuel_path),
                "files": None,
                "columns": len(self._read_schema(fuel_path)),
            },
            {
                "split": split,
                "resource": "flights",
                "path": self._hf_path(FLIGHTS_DIR_TEMPLATE.format(split=split)),
                "rows": None,
                "files": len(flight_files),
                "columns": len(self._read_schema(flight_files[0])) if flight_files else None,
            },
        ]

        return pl.DataFrame(rows)

    def get_usable_flight_ids(self, split: Split | str = DEFAULT_SPLIT) -> list[str]:
        """Return flight_ids present in both flightlist and trajectory files for the split.

        This is the effective sample size for any modeling or full-trajectory EDA.
        Train currently yields 10,000 (1,037 flightlist entries lack traj parquet).
        """
        split = self._validate_split(split)
        fl = self.get_flightlist(split)
        id_col = next((c for c in FLIGHT_ID_COLUMNS if c in fl.columns), None)
        if not id_col:
            self.logger.warning("No flight id column found in flightlist for %s", split)
            return []
        fl_ids = set(fl[id_col].cast(pl.Utf8).to_list())
        files = self.list_flight_files(split)
        traj_ids: set[str] = set()
        for p in files:
            stem = PurePosixPath(p).stem
            traj_ids.add(stem)
        usable = sorted(fl_ids & traj_ids)
        self.logger.info("Usable flights (flightlist + traj) for '%s': %d / %d", split, len(usable), len(fl_ids))
        return usable

    def _validate_split(self, split: Split | str) -> Split:
        """Validate and normalize a split value."""

        if split not in VALID_SPLITS:
            valid = ", ".join(VALID_SPLITS)
            raise ValueError(f"Invalid split '{split}'. Expected one of: {valid}.")
        return split  # type: ignore[return-value]

    def _hf_options(self) -> dict[str, str]:
        """Return Hugging Face filesystem options."""

        options: dict[str, str] = {}
        if self.token:
            options["token"] = self.token
        if self.endpoint:
            options["endpoint"] = self.endpoint
        return options

    def _storage_options(self) -> dict[str, str] | None:
        """Return storage options suitable for Polars."""

        options = self._hf_options()
        return options or None

    def _hf_path(self, relative_path: str) -> str:
        """Build a native ``hf://`` dataset path from a repo-relative path."""

        return f"{HF_DATASET_PREFIX}/{self.repo_id}/{relative_path.lstrip('/')}"

    def _hf_fs_path(self, path: str) -> str:
        """Build an ``HfFileSystem`` path from an ``hf://`` or relative path."""

        normalized = self._normalize_hf_path(path)
        prefix = f"{HF_DATASET_PREFIX}/"
        if normalized.startswith(prefix):
            return f"datasets/{normalized.removeprefix(prefix)}"
        return normalized.removeprefix("hf://")

    def _normalize_hf_path(self, path: str) -> str:
        """Normalize a full, filesystem, or repo-relative path to ``hf://``."""

        text = str(path).strip()
        if text.startswith("hf://"):
            return text

        normalized = text.replace("\\", "/").lstrip("/")
        dataset_prefix = f"datasets/{self.repo_id}/"
        repo_prefix = f"{self.repo_id}/"

        if normalized.startswith(dataset_prefix):
            return f"hf://{normalized}"
        if normalized.startswith(repo_prefix):
            return f"{HF_DATASET_PREFIX}/{normalized}"

        return self._hf_path(normalized)

    def _load_parquet(self, path: str) -> pl.DataFrame:
        """Read parquet into a Polars ``DataFrame``, preferring lazy scanning."""

        storage_options = self._storage_options()
        try:
            return pl.scan_parquet(path, storage_options=storage_options).collect()
        except Exception as lazy_exc:
            self.logger.debug("Lazy scan failed for %s; trying direct parquet read.", path, exc_info=True)
            try:
                return pl.read_parquet(path, storage_options=storage_options)
            except Exception:
                try:
                    with self.fs.open(self._hf_fs_path(path), mode="rb") as file:
                        return pl.read_parquet(file)
                except Exception as exc:  # pragma: no cover - depends on remote Hub state
                    self._raise_friendly_error(exc, path, cause=lazy_exc)

        raise RuntimeError(f"Unable to load parquet file: {path}")

    def _read_schema(self, path: str) -> dict[str, pl.DataType]:
        """Read a parquet schema without materializing the table rows."""

        storage_options = self._storage_options()
        try:
            lazy_frame = pl.scan_parquet(path, storage_options=storage_options)
            if hasattr(lazy_frame, "collect_schema"):
                return dict(lazy_frame.collect_schema())
            return dict(lazy_frame.schema)
        except Exception as scan_exc:
            self.logger.debug("Lazy schema read failed for %s; trying parquet metadata.", path, exc_info=True)
            try:
                with self.fs.open(self._hf_fs_path(path), mode="rb") as file:
                    arrow_schema = pq.ParquetFile(file).schema_arrow
                polars_schema = pl.from_arrow(arrow_schema.empty_table()).schema
                return {field.name: polars_schema[field.name] for field in arrow_schema}
            except Exception as exc:  # pragma: no cover - depends on remote Hub state
                self._raise_friendly_error(exc, path, cause=scan_exc)

        raise RuntimeError(f"Unable to read parquet schema: {path}")

    def _row_count(self, path: str) -> int:
        """Count parquet rows using a lazy aggregate."""

        storage_options = self._storage_options()
        count_expr = pl.len() if hasattr(pl, "len") else pl.count()
        try:
            result = pl.scan_parquet(path, storage_options=storage_options).select(count_expr.alias("rows")).collect()
        except Exception as scan_exc:
            self.logger.debug("Lazy row count failed for %s; trying parquet metadata.", path, exc_info=True)
            try:
                with self.fs.open(self._hf_fs_path(path), mode="rb") as file:
                    return int(pq.ParquetFile(file).metadata.num_rows)
            except Exception as exc:  # pragma: no cover - depends on remote Hub state
                self._raise_friendly_error(exc, path, cause=scan_exc)
        return int(result.item())

    def _flightlist_row_for_id(self, flight_id: str, split: Split) -> dict[str, Any] | None:
        """Return the first matching flightlist row for a flight id, if present."""

        path = self._hf_path(FLIGHTLIST_TEMPLATE.format(split=split))
        storage_options = self._storage_options()

        try:
            lazy_frame = pl.scan_parquet(path, storage_options=storage_options)
            schema = lazy_frame.collect_schema() if hasattr(lazy_frame, "collect_schema") else lazy_frame.schema
        except Exception:
            flightlist = self.get_flightlist(split=split)
            schema = flightlist.schema
            id_columns = [column for column in FLIGHT_ID_COLUMNS if column in schema]
            if not id_columns:
                self.logger.warning("No recognized flight id column found in %s", path)
                return None

            predicate = reduce(
                lambda left, right: left | right,
                (pl.col(column).cast(pl.Utf8) == flight_id for column in id_columns),
            )
            result = flightlist.lazy().filter(predicate).limit(1).collect()
            if result.is_empty():
                self.logger.info("Flight id '%s' was not found in %s", flight_id, path)
                return None
            return result.row(0, named=True)

        id_columns = [column for column in FLIGHT_ID_COLUMNS if column in schema]
        if not id_columns:
            self.logger.warning("No recognized flight id column found in %s", path)
            return None

        predicates = [pl.col(column).cast(pl.Utf8) == flight_id for column in id_columns]
        predicate = reduce(lambda left, right: left | right, predicates)

        try:
            result = lazy_frame.filter(predicate).limit(1).collect()
        except Exception as exc:  # pragma: no cover - depends on remote Hub state
            self._raise_friendly_error(exc, path)

        if result.is_empty():
            self.logger.info("Flight id '%s' was not found in %s", flight_id, path)
            return None
        return result.row(0, named=True)

    def _path_from_flightlist_row(self, row: dict[str, Any], split: Split) -> str | None:
        """Extract and normalize a flight parquet path from a flightlist row."""

        for column in FLIGHT_PATH_COLUMNS:
            value = row.get(column)
            if value is None or value == "":
                continue
            text = str(value)
            if text.startswith("hf://") or "/" in text or "\\" in text:
                return self._normalize_hf_path(text)
            return self._hf_path(f"{FLIGHTS_DIR_TEMPLATE.format(split=split)}/{text}")
        return None

    def _flight_id_path_candidates(self, flight_id: str, split: Split) -> list[str]:
        """Return common file path candidates for a flight id."""

        directory = FLIGHTS_DIR_TEMPLATE.format(split=split)
        safe_id = flight_id.strip().replace("\\", "_").replace("/", "_")
        names = (
            f"{safe_id}.parquet",
            f"flight_{safe_id}.parquet",
            f"flight_id={safe_id}.parquet",
        )
        return [self._hf_path(f"{directory}/{name}") for name in names]

    def _path_exists(self, path: str) -> bool:
        """Return whether a path exists on Hugging Face without reading it."""

        try:
            return bool(self.fs.exists(self._hf_fs_path(path)))
        except Exception as exc:  # pragma: no cover - depends on remote Hub state
            self.logger.debug("Could not check whether %s exists: %s", path, exc)
            return False

    def _matching_listed_flight_files(self, flight_id: str, split: Split) -> list[str]:
        """Find listed flight files whose stem matches or clearly contains an id."""

        files = self.list_flight_files(split=split)
        exact_matches: list[str] = []
        contains_matches: list[str] = []

        for path in files:
            stem = PurePosixPath(path).stem
            if stem == flight_id or stem == f"flight_{flight_id}":
                exact_matches.append(path)
            elif flight_id in stem:
                contains_matches.append(path)

        return exact_matches or contains_matches

    def _raise_friendly_error(self, exc: Exception, resource: str, cause: Exception | None = None) -> None:
        """Raise a clear local exception for common Hugging Face failures."""

        message = str(exc)
        chained = cause or exc
        cause_message = "" if cause is None else str(cause)
        combined_message = f"{message} {cause_message}".strip()
        lowered = combined_message.lower()

        auth_error = (
            isinstance(exc, LocalTokenNotFoundError)
            or "401" in combined_message
            or "403" in combined_message
            or "unauthorized" in lowered
            or "forbidden" in lowered
            or "authentication" in lowered
            or "invalid username or password" in lowered
        )
        not_found_error = (
            isinstance(exc, (FileNotFoundError, RepositoryNotFoundError))
            or "404" in combined_message
            or "not found" in lowered
            or "no such file" in lowered
        )
        ssl_error = "ssl" in lowered or "certificate verify failed" in lowered

        if auth_error:
            friendly = (
                f"Unable to access '{resource}'. The AeroTwin dataset is private; set HF_TOKEN "
                "or run 'huggingface-cli login' with an account that can access "
                f"'{self.repo_id}'."
            )
            self.logger.error(friendly)
            raise PermissionError(friendly) from chained

        if not_found_error:
            friendly = f"Could not find '{resource}' in Hugging Face dataset '{self.repo_id}'."
            self.logger.error(friendly)
            raise FileNotFoundError(friendly) from chained

        if ssl_error:
            friendly = (
                f"Could not verify the HTTPS certificate while accessing '{resource}'. "
                "Check your Python certificate store, proxy settings, or corporate CA configuration."
            )
            self.logger.error(friendly)
            raise ConnectionError(friendly) from chained

        if isinstance(exc, HfHubHTTPError):
            friendly = f"Hugging Face Hub request failed while accessing '{resource}': {message}"
            self.logger.error(friendly)
            raise RuntimeError(friendly) from chained

        friendly = f"Failed to access '{resource}': {message}"
        self.logger.error(friendly)
        raise RuntimeError(friendly) from chained


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")

    loader = AeroDataLoader()

    print("Dataset summary")
    print(loader.dataset_summary(split=DEFAULT_SPLIT))

    print("\nSchema")
    print(loader.get_schema(split=DEFAULT_SPLIT))

    print("\nRandom flight example")
    random_files = loader.sample_flight_files(n=1, split=DEFAULT_SPLIT, seed=DEFAULT_RANDOM_SEED)
    if random_files:
        random_path = random_files[0]
        print(f"Path: {random_path}")
        print(loader.load_flight(random_path).head())
    else:
        print(f"No flight files found for split '{DEFAULT_SPLIT}'.")

    print("\nUsable flights (flightlist + traj overlap)")
    usable = loader.get_usable_flight_ids(split=DEFAULT_SPLIT)
    print(f"Usable for {DEFAULT_SPLIT}: {len(usable)}")

    print("\nQuick ACARS air-data completeness (sample 3 usable)")
    import random
    random.seed(DEFAULT_RANDOM_SEED)
    sample_ids = random.sample(usable, min(3, len(usable))) if usable else []
    for fid in sample_ids:
        try:
            tr = loader.load_flight_by_id(fid, split=DEFAULT_SPLIT)
            ac = tr.filter(pl.col("source") == "acars") if "source" in tr.columns else pl.DataFrame()
            n_ac = len(ac)
            m_ok = int(ac["mach"].is_not_null().sum()) if n_ac else 0
            tas_ok = int(ac["TAS"].is_not_null().sum()) if n_ac else 0
            cas_ok = int(ac["CAS"].is_not_null().sum()) if n_ac else 0
            print(f"  {fid}: acars={n_ac} mach_ok={m_ok} TAS_ok={tas_ok} CAS_ok={cas_ok}")
        except Exception as exc:
            print(f"  {fid}: load error {exc}")
