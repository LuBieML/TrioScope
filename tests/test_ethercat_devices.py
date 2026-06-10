import unittest
from unittest.mock import patch

from src.ai import ethercat_devices as dev


class TestVendorName(unittest.TestCase):
    def test_known_vendor(self):
        self.assertEqual(dev.vendor_name(0x00000002), "Beckhoff Automation")
        self.assertEqual(dev.vendor_name(0x000002DE), "Trio Motion Technology")

    def test_unknown_vendor_shows_hex(self):
        self.assertEqual(dev.vendor_name(0x0BADF00D), "Unknown (0x0BADF00D)")

    def test_zero_vendor_is_empty(self):
        self.assertEqual(dev.vendor_name(0), "")

    def test_cached_registry_overrides_builtin(self):
        with patch.object(dev, "_online_vendors", {0x0BADF00D: "ACME Drives"}):
            self.assertEqual(dev.vendor_name(0x0BADF00D), "ACME Drives")


class TestProductLabel(unittest.TestCase):
    def test_trio_drive_type(self):
        self.assertEqual(dev.product_label(0x2DE, 0, drive_type=42), "DX4")
        self.assertEqual(dev.product_label(0x2DE, 0, drive_type=41), "DX3")

    def test_unknown_drive_type(self):
        self.assertEqual(dev.product_label(0, 0, drive_type=99), "T99")

    def test_beckhoff_coupler_decode(self):
        # EK1100 bus coupler
        self.assertEqual(dev.product_label(0x2, 0x044C2C52), "EK1100")

    def test_beckhoff_terminal_decode(self):
        # EL2002 output terminal
        self.assertEqual(dev.product_label(0x2, 0x07D23052), "EL2002")

    def test_profile_fallback_drive(self):
        self.assertEqual(dev.product_label(0x539, 0, device_type=0x00020192), "Drive")

    def test_profile_fallback_io(self):
        self.assertEqual(dev.product_label(0x44, 0, device_type=5001), "I/O")

    def test_hex_fallback(self):
        self.assertEqual(dev.product_label(0x539, 0x1234), "0x00001234")

    def test_empty_when_nothing_known(self):
        self.assertEqual(dev.product_label(0, 0), "")


class TestProfileAndRevision(unittest.TestCase):
    def test_profile_names(self):
        self.assertEqual(dev.device_profile_name(0x00020192), "Servo drive (CiA 402)")
        self.assertEqual(dev.device_profile_name(5001), "Modular device (MDP)")
        self.assertEqual(dev.device_profile_name(0), "")
        self.assertEqual(dev.device_profile_name(777), "Profile 777")

    def test_revision_str(self):
        self.assertEqual(dev.revision_str(0x00010002), "1.2")
        self.assertEqual(dev.revision_str(0), "")


class TestEtgHtmlParser(unittest.TestCase):
    def test_parses_hex_ids(self):
        page = """
        <table>
          <tr><th>Vendor ID</th><th>Company</th></tr>
          <tr><td>0x00000002</td><td>Beckhoff Automation GmbH &amp; Co. KG</td></tr>
          <tr><td>0x000002DE</td><td><b>Trio Motion Technology Ltd.</b></td></tr>
        </table>
        """
        vendors = dev.parse_etg_vendor_html(page)
        self.assertEqual(vendors[2], "Beckhoff Automation GmbH & Co. KG")
        self.assertEqual(vendors[0x2DE], "Trio Motion Technology Ltd.")

    def test_parses_decimal_ids(self):
        page = "<tr><td>1337</td><td>ACME Automation</td></tr>"
        vendors = dev.parse_etg_vendor_html(page)
        self.assertEqual(vendors[1337], "ACME Automation")

    def test_empty_page(self):
        self.assertEqual(dev.parse_etg_vendor_html("<html></html>"), {})


class TestWebSearchUrl(unittest.TestCase):
    def test_url_contains_vendor_and_product(self):
        url = dev.web_search_url(0x2, 0x044C2C52)
        self.assertIn("EtherCAT", url)
        self.assertIn("Beckhoff", url)
        self.assertIn("EK1100", url)

    def test_unknown_vendor_uses_hex_id(self):
        url = dev.web_search_url(0x0BADF00D, 0x1234)
        self.assertIn("0x0BADF00D", url)
        self.assertIn("0x00001234", url)


if __name__ == "__main__":
    unittest.main()
