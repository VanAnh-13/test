# ruff: noqa: BLE001, DTZ005, RUF013, SIM103
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

# Cấu hình logger cho module này
logger = logging.getLogger(__name__)


def normalize_param_grid(param_grid: dict | list[dict] | None) -> list[dict]:
    """
    Chuẩn hóa param_grid về định dạng list-of-dicts.

    Args:
        param_grid: dict đơn lẻ, list of dicts, hoặc None

    Returns:
        List of parameter dictionaries

    Raises:
        ValueError: Nếu định dạng không hợp lệ
    """
    if param_grid is None:
        return [{}]

    if isinstance(param_grid, dict):
        return [param_grid]

    if isinstance(param_grid, list):
        if len(param_grid) == 0:
            return [{}]
        if all(isinstance(d, dict) for d in param_grid):
            return param_grid
        raise ValueError("param_grid list chứa phần tử không phải dict")

    raise ValueError(
        f"param_grid phải là dict hoặc list of dicts, nhận được {type(param_grid)}"
    )


class SearchStrategy(ABC):
    """Base class for all search strategies."""

    def __init__(self, **kwargs):
        self.config = self.get_default_config()
        self.config.update(kwargs)
        self._search_start_time = None
        self._time_limit_reached = False

    # ==========================================================================
    # Timer Utilities cho Time Limit
    # ==========================================================================

    def _start_timer(self):
        """
        Bắt đầu đếm thời gian cho search.
        Gọi method này ở đầu hàm search.
        """
        self._search_start_time = time.time()
        self._time_limit_reached = False
        self._iteration_time_ema = None  # Reset EMA cho mỗi lần search mới

        max_time = self.config.get("max_time")
        if max_time is not None and self.config.get("verbose", 0) > 0:
            logger.info(f"Time limit: {max_time} giây")

    def _check_time_status(self) -> tuple[float | None, bool]:
        """
        Kiểm tra trạng thái thời gian của quá trình search.

        Returns:
            Tuple[Optional[float], bool]:
                - remaining_time: Thời gian còn lại (giây), None nếu không có time limit
                - is_exceeded: True nếu đã vượt quá time limit, False nếu chưa
        """
        max_time = self.config.get("max_time")

        # Nếu không có time limit
        if max_time is None:
            return None, False

        # Tính thời gian đã trôi qua
        elapsed = (
            0.0
            if self._search_start_time is None
            else time.time() - self._search_start_time
        )
        remaining = max(0.0, max_time - elapsed)

        # Kiểm tra đã vượt quá chưa
        is_exceeded = elapsed >= max_time
        if is_exceeded:
            self._time_limit_reached = True

        return remaining, is_exceeded

    def _should_start_next_iteration(self, iteration_duration: float = None) -> bool:
        """
        Kiểm tra xem có nên bắt đầu iteration tiếp theo không, dựa trên
        thời gian còn lại và ước tính thời gian mỗi iteration.

        Phương thức này sử dụng EMA (exponential moving average) để ước tính
        thời gian cho iteration tiếp theo. Nếu thời gian ước tính vượt quá
        thời gian còn lại, trả về False để dừng sớm (proactive stop).

        Args:
            iteration_duration: Thời gian iteration vừa hoàn thành (giây).
                              Nếu None, chỉ kiểm tra time exceeded.

        Returns:
            bool: True nếu nên tiếp tục, False nếu nên dừng.
        """
        remaining_time, is_exceeded = self._check_time_status()

        # Đã vượt quá time limit
        if is_exceeded:
            return False

        # Không có time limit
        if remaining_time is None:
            return True

        # Cập nhật EMA nếu có iteration_duration
        if iteration_duration is not None:
            if (
                not hasattr(self, "_iteration_time_ema")
                or self._iteration_time_ema is None
            ):
                self._iteration_time_ema = iteration_duration
            else:
                # EMA: 70% giá trị mới, 30% giá trị cũ
                self._iteration_time_ema = (
                    0.7 * iteration_duration + 0.3 * self._iteration_time_ema
                )

        # Kiểm tra proactive: ước tính iteration tiếp theo có vượt quá không
        if (
            hasattr(self, "_iteration_time_ema")
            and self._iteration_time_ema is not None
        ):
            # Nhân 1.2x safety factor (iteration tiếp có thể chậm hơn)
            estimated_next = self._iteration_time_ema * 1.2
            if estimated_next > remaining_time:
                logger.info(
                    f"Dừng proactive: ước tính iteration tiếp ~{estimated_next:.1f}s "
                    f"nhưng chỉ còn {remaining_time:.1f}s"
                )
                self._time_limit_reached = True
                return False

        return True

    def _should_apply_early_stopping(self) -> bool:
        """
        Xác định có nên áp dụng early stopping hay không.

        Logic:
        - Nếu max_time được set: KHÔNG áp dụng early stopping (ưu tiên time)
        - Nếu không có max_time: Áp dụng early stopping theo cấu hình

        Returns:
            bool: True nếu nên áp dụng early stopping, False nếu không
        """
        max_time = self.config.get("max_time")

        # Nếu có time limit, không áp dụng early stopping
        if max_time is not None:
            return False

        # Không có time limit, áp dụng early stopping theo config
        return True

    @staticmethod
    def _load_yaml_config(config_name: str) -> dict[str, Any]:
        """
        Tải cấu hình từ file YAML chuẩn ({config_name}_config.yml).

        Args:
            config_name: Tên config (vd: 'base', 'grid_search', 'bayesian_search')

        Returns:
            Dict chứa cấu hình từ file YAML
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, f"{config_name}_config.yml")
        default_config_file = os.path.join(
            current_dir, f"{config_name}_default_config.yml"
        )

        for filepath in (config_file, default_config_file):
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        loaded_config = yaml.safe_load(f) or {}
                        if loaded_config:
                            return loaded_config
                except Exception as e:
                    logger.warning(f"Không thể tải cấu hình từ {filepath}: {e}")

        return {}

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """Trả về cấu hình mặc định cho strategy này, đọc từ file YAML."""
        # Tải config từ file YAML
        yaml_config = SearchStrategy._load_yaml_config("base")

        # Tạo StratifiedKFold từ config
        cv_n_splits = yaml_config.get("cv_n_splits", 5)
        cv_shuffle = yaml_config.get("cv_shuffle", True)
        cv_random_state = yaml_config.get("cv_random_state", 42)
        cv = StratifiedKFold(
            n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=cv_random_state
        )

        # Config mặc định (fallback nếu YAML không có)
        config = {
            "cv": cv,
            "scoring": None,
            "metric_sort": yaml_config.get("metric_sort", "accuracy"),
            "n_jobs": yaml_config.get("n_jobs", -1),
            "verbose": yaml_config.get("verbose", 0),
            "random_state": yaml_config.get("random_state"),
            "error_score": yaml_config.get("error_score", "raise"),
            "log_dir": yaml_config.get("log_dir", "logs"),
            "save_log": yaml_config.get("save_log", False),
            "max_time": yaml_config.get("max_time", None),
        }

        return config

    # ==========================================================================
    # Shared CV-result Helpers (dùng chung cho mọi strategy)
    # ==========================================================================

    @staticmethod
    def _extract_cv_mean_scores(
        cv_out: dict[str, Any],
        scoring: dict[str, Any] | None,
        primary_metric: str,
    ) -> dict[str, float]:
        """Trích mean test score cho từng metric từ output của cross_validate.

        Args:
            cv_out: Dict trả về từ sklearn.model_selection.cross_validate
            scoring: Dict scoring config (None nếu cross_validate dùng metric đơn)
            primary_metric: Metric chính, dùng làm key khi scoring là None

        Returns:
            Dict {metric_name: mean_test_score}
        """
        all_scores: dict[str, float] = {}
        if scoring:
            for key in scoring:
                test_key = f"test_{key}"
                if test_key in cv_out:
                    all_scores[key] = float(np.mean(cv_out[test_key]))
        elif "test_score" in cv_out:
            all_scores[primary_metric] = float(np.mean(cv_out["test_score"]))
        return all_scores

    @staticmethod
    def _rank_descending(values: list[float]) -> list[int]:
        """Xếp hạng giảm dần theo điểm (1 = tốt nhất)."""
        return (np.argsort(np.argsort(-np.array(values))) + 1).tolist()

    @staticmethod
    def _init_cv_results(
        metric_names: list[str],
        extra_keys: list[str] | None = None,
        with_rank: bool = False,
    ) -> dict[str, Any]:
        """Khởi tạo dict cv_results_ với các key chuẩn và key theo metric.

        Args:
            metric_names: Danh sách metric cần key mean_test_/std_test_
            extra_keys: Các key bổ sung (vd: resource_frac, mean_fit_time)
            with_rank: True để tạo sẵn key rank_test_score/rank_test_{metric}
        """
        cv_results: dict[str, Any] = {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
        }
        for key in extra_keys or []:
            cv_results[key] = []
        if with_rank:
            cv_results["rank_test_score"] = []
        for metric in metric_names:
            cv_results[f"mean_test_{metric}"] = []
            cv_results[f"std_test_{metric}"] = []
            if with_rank:
                cv_results[f"rank_test_{metric}"] = []
        return cv_results

    @staticmethod
    def _append_cv_result(
        cv_results: dict[str, Any],
        params: dict[str, Any],
        score: float,
        all_scores: dict[str, float],
        metric_names: list[str],
    ) -> None:
        """Thêm một kết quả đánh giá vào cv_results_ (std mặc định 0.0)."""
        cv_results["params"].append(params)
        cv_results["mean_test_score"].append(score)
        cv_results["std_test_score"].append(0.0)
        for metric in metric_names:
            cv_results[f"mean_test_{metric}"].append(all_scores.get(metric, 0.0))
            cv_results[f"std_test_{metric}"].append(0.0)

    @abstractmethod
    def search(
        self,
        model: BaseEstimator,
        param_grid: list[dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """Thực thi thuật toán tìm kiếm.

        Args:
            model: Mô hình scikit-learn cần tối ưu hóa
            param_grid: List of dicts, mỗi dict chứa các tham số cần tìm kiếm.
                        Ví dụ: [{'kernel': ['rbf'], 'C': [1, 10]}, {'kernel': ['linear'], 'C': [1, 10]}]
            X: Dữ liệu features
            y: Dữ liệu target
            **kwargs: Các tham số bổ sung

        Returns:
            tuple: (best_params, best_score, best_all_scores, cv_results, time_limit_reached)
                time_limit_reached (bool): True nếu search bị dừng do hết thời gian
        """

    def set_config(self, **kwargs):
        """Cập nhật cấu hình"""
        self.config.update(kwargs)
        return self

    def create_log_file_path(
        self, model: BaseEstimator, strategy_name: str = ""
    ) -> str | None:
        """Tạo đường dẫn file log để lưu kết quả tìm kiếm.

        Phương thức này tạo đường dẫn file log chuẩn hóa dựa trên cấu hình.
        Nó đảm bảo thư mục log tồn tại và tạo tên file có timestamp.

        Args:
            model: Mô hình đang được sử dụng cho tìm kiếm
            strategy_name: Tên tùy chọn cho strategy (mặc định là tên class)

        Returns:
            str: Đường dẫn đến file log nếu save_log là True, None nếu ngược lại
        """
        if not self.config.get("save_log", False):
            return None

        log_dir = self.config.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.__class__.__name__

        # Sử dụng tên strategy được cung cấp hoặc tạo từ tên class
        if strategy_name is None:
            # Chuyển đổi tên class từ CamelCase sang snake_case
            class_name = self.__class__.__name__
            # Loại bỏ hậu tố 'Strategy' nếu có
            class_name = class_name.removesuffix("Strategy")
            class_name = class_name.removesuffix("SearchStrategy")
            # Chuyển đổi sang snake_case
            import re

            strategy_name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()

        log_file = os.path.join(
            log_dir, f"{strategy_name}_{model_name}_{timestamp}.csv"
        )
        return log_file

    @staticmethod
    def convert_numpy_types(obj: Any) -> Any:
        """Chuyển đổi kiểu numpy sang kiểu Python gốc một cách đệ quy.

        Điều này quan trọng cho JSON serialization và tránh
        lỗi kiểu không thể hash.

        Args:
            obj: Đối tượng cần chuyển đổi (có thể là dict, list, scalar, v.v.)

        Returns:
            Đối tượng với tất cả kiểu numpy đã được chuyển đổi sang kiểu Python gốc
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "item"):
            return obj.item()
        elif isinstance(obj, dict):
            return {
                key: SearchStrategy.convert_numpy_types(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [SearchStrategy.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(SearchStrategy.convert_numpy_types(item) for item in obj)
        else:
            return obj

    def _finalize_results(
        self,
        best_params: dict[str, Any],
        best_score: float,
        best_all_scores: dict[str, float],
        cv_results: dict[str, Any],
    ) -> tuple:
        """Xóa cache và chuyển đổi kiểu numpy trước khi trả về kết quả.

        Phương thức này nên được gọi ở cuối phương thức search để:
        1. Xóa tất cả cache để giải phóng bộ nhớ
        2. Chuyển đổi kiểu numpy sang kiểu Python gốc để serialization

        Args:
            best_params: Tham số tốt nhất tìm được
            best_score: Điểm số tốt nhất đạt được
            best_all_scores: Tất cả điểm số metric cho tham số tốt nhất
            cv_results: Kết quả cross-validation chi tiết

        Returns:
            tuple: (best_params, best_score, best_all_scores, cv_results, time_limit_reached)
                Tất cả kiểu numpy đã chuyển đổi sang Python gốc.
                time_limit_reached (bool): True nếu search bị dừng do hết thời gian
        """
        # Xóa cache sau khi tìm kiếm hoàn thành
        if hasattr(self, "_decode_cache"):
            self._decode_cache.clear()
        if hasattr(self, "_evaluation_cache"):
            self._evaluation_cache.clear()
        if hasattr(self, "_model_copies"):
            self._model_copies.clear()

        # Chuyển đổi tất cả kiểu numpy sang kiểu Python gốc
        best_params = self.convert_numpy_types(best_params)
        best_score = self.convert_numpy_types(best_score)
        best_all_scores = self.convert_numpy_types(best_all_scores)
        cv_results = self.convert_numpy_types(cv_results)

        return (
            best_params,
            best_score,
            best_all_scores,
            cv_results,
            self._time_limit_reached,
        )

    def _init_search_log(self):
        """Khởi tạo danh sách log cho quá trình tìm kiếm."""
        self._search_log = []

    def _log_evaluation(
        self,
        model_name: str,
        strategy_name: str,
        params: dict[str, Any],
        scores: dict[str, float],
        iteration: int = 0,
        total: int = 0,
    ):
        """Ghi log kết quả đánh giá một tổ hợp tham số ra console và lưu vào danh sách."""
        if not hasattr(self, "_search_log"):
            self._search_log = []

        record: dict[str, Any] = {
            "model": model_name,
            "run_type": strategy_name,
            "best_params": str(params),
        }
        record.update(scores)
        self._search_log.append(record)

        if self.config.get("verbose", 0) > 0:
            progress = (
                f"[{iteration}/{total}] "
                if iteration is not None and total is not None
                else ""
            )
            scores_str = ", ".join([f"{k}={v:.4f}" for k, v in scores.items()])
            logger.info(
                f"{progress}{model_name} | {strategy_name} | {params} | {scores_str}"
            )

    def _save_search_log(self, log_file: str | None, silent: bool = False):
        """Lưu log tìm kiếm vào file CSV theo format chuẩn."""
        if not log_file or not hasattr(self, "_search_log") or not self._search_log:
            return

        import pandas as pd

        df = pd.DataFrame(self._search_log)
        df.to_csv(log_file, index=False)
        if not silent:
            logger.info(f"Log tìm kiếm đã lưu vào: {log_file}")
