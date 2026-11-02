from types import SimpleNamespace

import pytest

from mlrescurix_worker.scientific import kaggle_output_transport as transport


class Request:
    pass


class Response:
    status_code = 200
    headers = {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, size): return iter([b"payload"])


class Requests:
    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"]["Accept-Encoding"] == "identity"
        return Response()


def client(*items, next_page_token=""):
    page = SimpleNamespace(files=items, next_page_token=next_page_token)
    api = SimpleNamespace(list_kernel_session_output=lambda request: page)
    return SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=api))


def item(name="weights.bin", url="https://storage.googleapis.com/bucket/file"):
    return SimpleNamespace(file_name=name, url=url)


def test_download_streams_bounded_flat_files(tmp_path):
    transport.fetch_outputs(client(item()), Request, Requests(), "owner", "kernel", tmp_path)
    assert (tmp_path / "weights.bin").read_bytes() == b"payload"


@pytest.mark.parametrize("bad", [item("../escape"), item("/absolute"), item("nested/file"),
                                 item(url="http://storage.googleapis.com/bucket/file"),
                                 item(url="https://storage.googleapis.com.evil.test/file"),
                                 item(url="https://user@storage.googleapis.com/file")])
def test_download_rejects_unsafe_remote_names_and_urls(tmp_path, bad):
    with pytest.raises(ValueError):
        transport.fetch_outputs(client(bad), Request, Requests(), "owner", "kernel", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_duplicates_and_oversized_stream(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="duplicate"):
        transport.fetch_outputs(client(item(), item()), Request, Requests(), "owner", "kernel", tmp_path)
    (tmp_path / "weights.bin").unlink()
    monkeypatch.setattr(transport, "MAX_BYTES", 3)
    with pytest.raises(ValueError, match="byte limit"):
        transport.fetch_outputs(client(item()), Request, Requests(), "owner", "kernel", tmp_path)
    assert (tmp_path / "weights.bin").stat().st_size <= 3


def test_download_refuses_redirects_and_nonadvancing_pagination(tmp_path):
    class Redirect(Response):
        status_code = 302
    class RedirectRequests:
        def get(self, *args, **kwargs): return Redirect()
    with pytest.raises(ValueError, match="redirect"):
        transport.fetch_outputs(client(item()), Request, RedirectRequests(), "owner", "kernel", tmp_path)
    with pytest.raises(ValueError, match="pagination"):
        transport.fetch_outputs(client(next_page_token="repeat"), Request, Requests(), "owner", "kernel", tmp_path)


def test_installed_kaggle_sdk_request_contract(tmp_path):
    sdk = pytest.importorskip("kagglesdk.kernels.types.kernels_api_service")
    observed = []
    def listing(request):
        observed.append((request.user_name, request.kernel_slug, request.page_size))
        return SimpleNamespace(files=[item()], next_page_token="")
    api = SimpleNamespace(list_kernel_session_output=listing)
    fake_client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=api))
    transport.fetch_outputs(fake_client, sdk.ApiListKernelSessionOutputRequest, Requests(), "owner", "kernel", tmp_path)
    assert observed == [("owner", "kernel", 100)]


def test_download_refuses_compressed_output(tmp_path):
    class Compressed(Response):
        headers = {"Content-Encoding": "gzip"}
    class CompressedRequests:
        def get(self, *args, **kwargs): return Compressed()
    with pytest.raises(ValueError, match="compressed"):
        transport.fetch_outputs(client(item()), Request, CompressedRequests(), "owner", "kernel", tmp_path)
